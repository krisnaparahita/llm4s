package org.llm4s.vectorstore

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec

/**
 * Tests for HybridSearcher.Config builders and the strategy HybridSearcher(config) is built with.
 */
class HybridSearcherConfigSpec extends AnyWordSpec with Matchers {

  private val docs = Seq(
    ("scala-guide", "Scala is a powerful functional programming language", Array(0.9f, 0.1f, 0.0f)),
    ("scala-jvm", "Scala runs on the JVM and interoperates with Java", Array(0.8f, 0.2f, 0.1f)),
    ("python-intro", "Python is a dynamically typed programming language", Array(0.2f, 0.9f, 0.0f)),
    ("java-basics", "Java is an object-oriented programming language", Array(0.3f, 0.8f, 0.2f)),
    ("rust-systems", "Rust is a systems programming language focused on safety", Array(0.1f, 0.3f, 0.9f))
  )

  private def withSearcher[A](config: HybridSearcher.Config)(f: HybridSearcher => A): A = {
    val searcher = HybridSearcher(config).fold(e => fail(s"Failed: ${e.formatted}"), identity)
    try f(searcher)
    finally searcher.close()
  }

  "HybridSearcher.Config" should {

    "default to the default fusion strategy" in {
      HybridSearcher.Config.default.defaultStrategy shouldBe FusionStrategy.default
    }

    "use in-memory stores for Config.inMemory" in {
      val config = HybridSearcher.Config.inMemory
      config.vectorStoreConfig shouldBe VectorStoreFactory.Config.inMemory
      config.keywordIndexConfig shouldBe SQLiteKeywordIndex.Config.inMemory
      config.defaultStrategy shouldBe FusionStrategy.default
    }

    "set RRF with k = 60 from withRRF()" in {
      val base = HybridSearcher.Config.inMemory.withStrategy(FusionStrategy.VectorOnly)
      base.withRRF() shouldBe base.copy(defaultStrategy = FusionStrategy.RRF(60))
    }

    "set RRF with the given k from withRRF(k)" in {
      val base = HybridSearcher.Config.inMemory
      base.withRRF(10) shouldBe base.copy(defaultStrategy = FusionStrategy.RRF(10))
    }

    "set the weights from withWeightedScore" in {
      val base = HybridSearcher.Config.inMemory
      base.withWeightedScore(0.7, 0.3) shouldBe base.copy(defaultStrategy = FusionStrategy.WeightedScore(0.7, 0.3))
    }

    "set only the strategy from withStrategy" in {
      val base = HybridSearcher.Config.inMemory
      base.withStrategy(FusionStrategy.KeywordOnly) shouldBe base.copy(defaultStrategy = FusionStrategy.KeywordOnly)
    }

    "set only the vector store config from withVectorStore" in {
      val base        = HybridSearcher.Config.inMemory.withRRF(10)
      val vectorStore = VectorStoreFactory.Config.sqlite("/tmp/vectors.db")
      base.withVectorStore(vectorStore) shouldBe base.copy(vectorStoreConfig = vectorStore)
    }

    "set only the keyword index config from withKeywordIndex" in {
      val base         = HybridSearcher.Config.inMemory.withRRF(10)
      val keywordIndex = SQLiteKeywordIndex.Config.inMemory.withTableName("other_docs")
      base.withKeywordIndex(keywordIndex) shouldBe base.copy(keywordIndexConfig = keywordIndex)
    }
  }

  "HybridSearcher(config)" should {

    "build a searcher with the configured default strategy" in {
      withSearcher(HybridSearcher.Config.inMemory.withWeightedScore(0.7, 0.3)) { searcher =>
        searcher.defaultStrategy shouldBe FusionStrategy.WeightedScore(0.7, 0.3)
      }
    }

    "build a searcher with the configured RRF k" in {
      withSearcher(HybridSearcher.Config.inMemory.withRRF(10)) { searcher =>
        searcher.defaultStrategy shouldBe FusionStrategy.RRF(10)
      }
    }

    "use the configured strategy when search is called without one" in {
      withSearcher(HybridSearcher.Config.inMemory.withStrategy(FusionStrategy.VectorOnly)) { searcher =>
        searcher.defaultStrategy shouldBe FusionStrategy.VectorOnly
        docs.foreach { case (id, content, embedding) =>
          searcher.vectorStore.upsert(VectorRecord(id, embedding, Some(content), Map("type" -> "doc")))
          searcher.keywordIndex.index(KeywordDocument(id, content, Map("type" -> "doc")))
        }

        val results = searcher
          .search(Array(0.85f, 0.15f, 0.0f), "Scala programming language", topK = 2)
          .fold(e => fail(s"Failed: ${e.formatted}"), identity)

        results should have size 2
        results.foreach { r =>
          r.keywordScore shouldBe None
          r.vectorScore shouldBe defined
        }
      }
    }
  }
}
