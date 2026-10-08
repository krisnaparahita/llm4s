---
layout: page
title: RAG Query Transformers
parent: User Guide
nav_order: 6
---

# RAG Query Transformers
{: .no_toc }

Rewrite the user's query before it is embedded, so retrieval works on a better question than the one that was typed.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Why rewrite queries?

Users write vague, conversational or abbreviated queries ("tell me about that config thing"). An embedding of that text sits far from the passage that answers it. A **query transformer** runs before retrieval and turns the raw query into something closer to how your documents are written: expand an abbreviation, resolve a vague reference, drop filler words.

The transformed query is the one that is embedded and searched, for both vector search and keyword search. The original is not used afterwards.

## The `QueryTransformer` trait

All transformers live in `org.llm4s.rag.transform`.

```scala
trait QueryTransformer {
  def transform(query: String): Result[String]
  def name: String
}
```

- `transform` returns the new query, or a `Left` error.
- `name` is a human-readable label used in logging and tracing.

## Built-in transformers

| Transformer | What it does |
|---|---|
| `LLMQueryRewriter(llmClient)` | Asks the model to rewrite the query into a precise, search-friendly one. Runs at temperature `0.0`. A failed model call becomes a `ProcessingError` with operation `query-rewrite`. |
| `LLMQueryRewriter(llmClient, systemPrompt)` | The same, with your own system prompt instead of `LLMQueryRewriter.DefaultSystemPrompt`. |
| `IdentityTransformer()` | Returns the query unchanged. Useful for testing a chain or as a placeholder. |

`LLMQueryRewriter` makes one extra model call per query, so it adds latency and cost to every search. Use a small, fast model for the client you pass to it.

## Adding transformers to a `RAGConfig`

`RAG.builder()` returns a `RAGConfig`. Add one transformer with `withQueryTransformer`, or set the whole list with `withQueryTransformers`:

```scala
import org.llm4s.rag.RAG
import org.llm4s.rag.transform.{ IdentityTransformer, LLMQueryRewriter }

// One transformer
val config = RAG.builder()
  .withEmbeddings("openai")
  .withQueryTransformer(LLMQueryRewriter(llmClient))

// Several at once, replacing any set earlier
val chained = RAG.builder()
  .withEmbeddings("openai")
  .withQueryTransformers(Seq(IdentityTransformer(), LLMQueryRewriter(llmClient)))
```

`withQueryTransformer` appends to the list; `withQueryTransformers` replaces it. Build the `RAG` from the config as described in the [Vector Store guide](vector-store); every call to `rag.query(...)` then applies the transformers first. With no transformers configured, the query is used as typed.

## Chaining and error behaviour

Transformers run in the order they were added, and each receives the output of the previous one. `QueryTransformer.applyChain` does the work:

```scala
QueryTransformer.applyChain("start", Seq(a, b, c)) // c(b(a("start")))
```

The chain **stops at the first error**: later transformers do not run, and `rag.query` returns that `Left`. There is no fallback to the original query. If a rewrite failing should not fail the search, wrap the transformer so it returns the input on error (see below).

## Writing your own transformer

Implement the trait. This one is deterministic: it expands abbreviations your users type but your documents spell out.

```scala
import org.llm4s.rag.transform.QueryTransformer
import org.llm4s.types.Result

final class AbbreviationExpander(expansions: Map[String, String]) extends QueryTransformer {

  override val name: String = "abbreviation-expander"

  override def transform(query: String): Result[String] =
    Right(
      query
        .split("\\s+")
        .map(word => expansions.getOrElse(word.toLowerCase, word))
        .mkString(" ")
    )
}
```

Call it directly, or run it through a chain:

```scala
val expander = new AbbreviationExpander(Map("db" -> "database", "cfg" -> "configuration"))

expander.transform("db cfg options")
// Right("database configuration options")

QueryTransformer.applyChain("db cfg options", Seq(expander, IdentityTransformer()))
// Right("database configuration options")
```

To make a transformer best-effort, catch the failure and return the input:

```scala
final class BestEffort(inner: QueryTransformer) extends QueryTransformer {
  override val name: String = s"best-effort(${inner.name})"
  override def transform(query: String): Result[String] =
    Right(inner.transform(query).getOrElse(query))
}
```

## Where the behaviour is tested

The order, the stop-at-first-error rule and the error type described above match these specs in `modules/rag/src/test/scala/org/llm4s/rag/transform/`:

- `QueryTransformerSpec` - sequential application and short-circuiting
- `LLMQueryRewriterSpec` - default prompt, temperature, `ProcessingError` on model failure
- `IdentityTransformerSpec` - pass-through

## Next steps

- [Vector Store](vector-store) - building a `RAG` pipeline
- [RAG Evaluation](rag-evaluation) - measure whether a transformer actually improves retrieval
