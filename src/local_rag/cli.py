"""CLI entry point for local-rag."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import (
    CHAT_MODEL_DEFAULT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_DIR,
    EMBED_MODEL_DEFAULT,
    OLLAMA_DEFAULT_URL,
    TOP_K,
)

console = Console()
err = Console(stderr=True)


@click.group()
def main() -> None:
    """local-rag — Ask questions about your documents using local LLMs.

    \b
    Quick start:
      rag add report.pdf
      rag ask "What are the key findings?"
      rag list
    """


@main.command("add")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--embed-model", default=EMBED_MODEL_DEFAULT, show_default=True)
@click.option("--host", default=OLLAMA_DEFAULT_URL, show_default=True)
@click.option("--chunk-size", default=CHUNK_SIZE, show_default=True)
@click.option("--chunk-overlap", default=CHUNK_OVERLAP, show_default=True)
def add_cmd(
    paths: tuple[Path, ...],
    embed_model: str,
    host: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Ingest documents into the local vector store.

    \b
    Supported formats: .txt, .md, .rst, .pdf, .docx
    """
    from .embedder import EmbedError, embed_texts
    from .loader import LoadError, chunk_text, load_document
    from .store import add_chunks

    for path in paths:
        console.print(f"[dim]Loading[/dim] {path.name}…")
        try:
            text = load_document(path)
        except LoadError as e:
            err.print(f"[red]Load error ({path.name}):[/red] {e}")
            continue

        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            err.print(f"[yellow]No text found in {path.name}, skipping.[/yellow]")
            continue

        console.print(f"  [dim]{len(chunks)} chunks → embedding with[/dim] {embed_model}…")
        try:
            with console.status("[bold green]Embedding…"):
                embeddings = embed_texts(chunks, model=embed_model, base_url=host)
        except EmbedError as e:
            err.print(f"[red]Embed error:[/red] {e}")
            sys.exit(1)

        added = add_chunks(chunks, embeddings, source=str(path.resolve()))
        console.print(
            f"  [green]Added {added} new chunks[/green]"
            + (f" (skipped {len(chunks) - added} duplicates)" if added < len(chunks) else "")
        )


@main.command("ask")
@click.argument("question")
@click.option("--embed-model", default=EMBED_MODEL_DEFAULT, show_default=True)
@click.option("--chat-model", default=CHAT_MODEL_DEFAULT, show_default=True)
@click.option("--host", default=OLLAMA_DEFAULT_URL, show_default=True)
@click.option("--top-k", default=TOP_K, show_default=True, help="Number of chunks to retrieve.")
@click.option("--source", default=None, help="Restrict retrieval to a specific source file.")
@click.option("--show-sources", is_flag=True, help="Print retrieved chunks below the answer.")
def ask_cmd(
    question: str,
    embed_model: str,
    chat_model: str,
    host: str,
    top_k: int,
    source: str | None,
    show_sources: bool,
) -> None:
    """Ask a question about your ingested documents."""
    from .embedder import EmbedError, embed_texts
    from .llm import answer
    from .store import list_sources, query

    sources = list_sources()
    if not sources:
        err.print("[yellow]No documents ingested yet.[/yellow] Run `rag add <file>` first.")
        sys.exit(1)

    try:
        with console.status("[bold green]Embedding question…"):
            [q_emb] = embed_texts([question], model=embed_model, base_url=host)
    except EmbedError as e:
        err.print(f"[red]Embed error:[/red] {e}")
        sys.exit(1)

    chunks = query(q_emb, top_k=top_k, source_filter=source)
    if not chunks:
        err.print("[yellow]No relevant chunks found.[/yellow]")
        sys.exit(1)

    try:
        with console.status(f"[bold green]Querying {chat_model}…"):
            response = answer(question, chunks, model=chat_model, base_url=host)
    except (ConnectionError, RuntimeError) as e:
        err.print(f"[red]LLM error:[/red] {e}")
        sys.exit(1)

    console.print(Panel(Markdown(response), title="Answer", border_style="green"))

    if show_sources:
        table = Table(title="Retrieved chunks", show_lines=True)
        table.add_column("Source", style="dim", max_width=40)
        table.add_column("Distance", justify="right")
        table.add_column("Excerpt")
        for c in chunks:
            table.add_row(
                Path(c["source"]).name,
                f"{c['distance']:.3f}",
                c["text"][:120] + "…" if len(c["text"]) > 120 else c["text"],
            )
        console.print(table)


@main.command("list")
def list_cmd() -> None:
    """List all ingested documents."""
    from .store import list_sources

    sources = list_sources()
    if not sources:
        console.print("[yellow]No documents ingested yet.[/yellow]")
        return

    console.print(f"[bold]{len(sources)} document(s) in store:[/bold]")
    for s in sources:
        console.print(f"  [cyan]{s}[/cyan]")


@main.command("remove")
@click.argument("source")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def remove_cmd(source: str, yes: bool) -> None:
    """Remove a document from the store by its path."""
    from rich.prompt import Confirm

    from .store import delete_source, list_sources

    sources = list_sources()
    # Support partial match
    matches = [s for s in sources if source in s]
    if not matches:
        err.print(f"[yellow]No document matching '{source}' found.[/yellow]")
        sys.exit(1)
    if len(matches) > 1:
        err.print(f"[yellow]Ambiguous match — found {len(matches)} documents:[/yellow]")
        for m in matches:
            err.print(f"  {m}")
        sys.exit(1)

    full_source = matches[0]
    if not yes and not Confirm.ask(f"Remove '{Path(full_source).name}' from store?"):
        console.print("[dim]Aborted.[/dim]")
        sys.exit(0)

    deleted = delete_source(full_source)
    console.print(f"[green]Removed {deleted} chunks for[/green] {Path(full_source).name}")


@main.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def clear_cmd(yes: bool) -> None:
    """Remove ALL documents from the store."""
    import shutil

    from rich.prompt import Confirm

    if not yes and not Confirm.ask(
        "[red]Remove ALL documents from the store?[/red]", default=False
    ):
        console.print("[dim]Aborted.[/dim]")
        sys.exit(0)

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        console.print("[green]Store cleared.[/green]")
    else:
        console.print("[yellow]Store was already empty.[/yellow]")
