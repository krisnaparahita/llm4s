package org.llm4s.agent.guardrails.rag

import org.llm4s.agent.guardrails.builtin.{ InjectionSensitivity, PIIDetector, PIIMasker, PromptInjectionDetector }
import org.llm4s.agent.guardrails.patterns.PIIPatterns.PIIType
import org.llm4s.agent.guardrails.{ GuardrailAction, InputGuardrail, OutputGuardrail }
import org.llm4s.error.NetworkError
import org.llm4s.llmconnect.LLMClient
import org.llm4s.llmconnect.model._
import org.llm4s.types.Result
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
 * Pins the exact guardrail types, order, thresholds and onFail settings of the
 * [[RAGGuardrails]] domain presets. No model calls are made: the stub client is
 * only passed through to the LLM-as-judge guardrails.
 */
class RAGGuardrailsPresetsSpec extends AnyFlatSpec with Matchers {

  private class StubLLMClient extends LLMClient {
    override def complete(conversation: Conversation, options: CompletionOptions): Result[Completion] =
      Left(NetworkError("Stub client: presets must not call the model", None, "stub://none"))

    override def streamComplete(
      conversation: Conversation,
      options: CompletionOptions,
      onChunk: StreamedChunk => Unit
    ): Result[Completion] = complete(conversation, options)

    override def getContextWindow(): Int     = 4096
    override def getReserveCompletion(): Int = 1024
  }

  private val client = new StubLLMClient
  private val topics = Seq("alpha", "beta")

  private def piiDetector(g: InputGuardrail)        = g.asInstanceOf[PIIDetector]
  private def injection(g: InputGuardrail)          = g.asInstanceOf[PromptInjectionDetector]
  private def topic(g: InputGuardrail)              = g.asInstanceOf[TopicBoundaryGuardrail]
  private def grounding(g: RAGGuardrail)            = g.asInstanceOf[GroundingGuardrail]
  private def relevance(g: RAGGuardrail)            = g.asInstanceOf[ContextRelevanceGuardrail]
  private def attribution(g: RAGGuardrail)          = g.asInstanceOf[SourceAttributionGuardrail]
  private def masker(g: OutputGuardrail): PIIMasker = g.asInstanceOf[PIIMasker]

  "RAGGuardrails.standardWithTopics" should "add a balanced topic boundary after the standard input guardrails" in {
    val config = RAGGuardrails.standardWithTopics(client, topics)

    config.inputGuardrails.map(_.getClass) shouldBe Seq(
      classOf[PIIDetector],
      classOf[PromptInjectionDetector],
      classOf[TopicBoundaryGuardrail]
    )
    injection(config.inputGuardrails(1)).sensitivity shouldBe InjectionSensitivity.Medium
    val t = topic(config.inputGuardrails(2))
    t.allowedTopics shouldBe topics
    t.threshold shouldBe 0.5
    t.onFail shouldBe GuardrailAction.Block

    config.outputGuardrails.map(_.getClass) shouldBe Seq(classOf[PIIMasker])
    config.ragGuardrails.map(_.getClass) shouldBe Seq(
      classOf[GroundingGuardrail],
      classOf[ContextRelevanceGuardrail]
    )
    grounding(config.ragGuardrails.head).threshold shouldBe 0.7
    grounding(config.ragGuardrails.head).strictMode shouldBe false
    relevance(config.ragGuardrails(1)).threshold shouldBe 0.5
    relevance(config.ragGuardrails(1)).minRelevantRatio shouldBe 0.5
    RAGGuardrails.allGuardrails(
      config
    ) should have size (config.inputGuardrails.size + config.outputGuardrails.size + config.ragGuardrails.size)
  }

  "RAGGuardrails.customerSupport" should "append the support topics to the product topics" in {
    val config = RAGGuardrails.customerSupport(client, topics)

    config.inputGuardrails.map(_.getClass) shouldBe Seq(
      classOf[PIIDetector],
      classOf[PromptInjectionDetector],
      classOf[TopicBoundaryGuardrail]
    )
    piiDetector(config.inputGuardrails.head).onFail shouldBe GuardrailAction.Block
    injection(config.inputGuardrails(1)).sensitivity shouldBe InjectionSensitivity.Medium
    val t = topic(config.inputGuardrails(2))
    t.allowedTopics.take(topics.size) shouldBe topics
    (t.allowedTopics should contain).allOf(
      "product questions",
      "technical support",
      "troubleshooting",
      "account issues",
      "billing questions",
      "feature requests"
    )
    t.threshold shouldBe 0.5
    t.onFail shouldBe GuardrailAction.Block

    config.outputGuardrails.map(_.getClass) shouldBe Seq(classOf[PIIMasker])
    config.ragGuardrails.map(_.getClass) shouldBe Seq(
      classOf[GroundingGuardrail],
      classOf[ContextRelevanceGuardrail]
    )
    grounding(config.ragGuardrails.head).threshold shouldBe 0.7
    grounding(config.ragGuardrails.head).onFail shouldBe GuardrailAction.Block
    RAGGuardrails.allGuardrails(
      config
    ) should have size (config.inputGuardrails.size + config.outputGuardrails.size + config.ragGuardrails.size)
  }

  "RAGGuardrails.softwareDocumentation" should "have no PII or output guardrails and require source attribution" in {
    val config = RAGGuardrails.softwareDocumentation(client)

    config.inputGuardrails.map(_.getClass) shouldBe Seq(
      classOf[PromptInjectionDetector],
      classOf[TopicBoundaryGuardrail]
    )
    injection(config.inputGuardrails.head).sensitivity shouldBe InjectionSensitivity.Medium
    val t = topic(config.inputGuardrails(1))
    (t.allowedTopics should contain).allOf("software development", "programming", "debugging", "APIs")
    t.threshold shouldBe 0.5

    config.outputGuardrails shouldBe empty
    config.ragGuardrails.map(_.getClass) shouldBe Seq(
      classOf[GroundingGuardrail],
      classOf[SourceAttributionGuardrail]
    )
    grounding(config.ragGuardrails.head).threshold shouldBe 0.7
    val a = attribution(config.ragGuardrails(1))
    a.requireAttributions shouldBe true
    a.minAttributionScore shouldBe 0.5
    a.onFail shouldBe GuardrailAction.Block
    RAGGuardrails.allGuardrails(
      config
    ) should have size (config.inputGuardrails.size + config.outputGuardrails.size + config.ragGuardrails.size)
  }

  "RAGGuardrails.research" should "use strict grounding and attribution but only prompt-injection input checks" in {
    val config = RAGGuardrails.research(client)

    config.inputGuardrails.map(_.getClass) shouldBe Seq(classOf[PromptInjectionDetector])
    injection(config.inputGuardrails.head).sensitivity shouldBe InjectionSensitivity.Medium
    config.outputGuardrails.map(_.getClass) shouldBe Seq(classOf[PIIMasker])
    config.ragGuardrails.map(_.getClass) shouldBe Seq(
      classOf[GroundingGuardrail],
      classOf[ContextRelevanceGuardrail],
      classOf[SourceAttributionGuardrail]
    )

    val g = grounding(config.ragGuardrails.head)
    g.threshold shouldBe 0.9
    g.strictMode shouldBe true
    val r = relevance(config.ragGuardrails(1))
    r.threshold shouldBe 0.5
    r.minRelevantRatio shouldBe 0.5
    val a = attribution(config.ragGuardrails(2))
    a.requireAttributions shouldBe true
    a.minAttributionScore shouldBe 0.8
    RAGGuardrails.allGuardrails(
      config
    ) should have size (config.inputGuardrails.size + config.outputGuardrails.size + config.ragGuardrails.size)
  }

  "RAGGuardrails.financial" should "use financial PII handling, strict topics and strict grounding" in {
    val config = RAGGuardrails.financial(client, topics)

    config.inputGuardrails.map(_.getClass) shouldBe Seq(
      classOf[PIIDetector],
      classOf[PromptInjectionDetector],
      classOf[TopicBoundaryGuardrail]
    )
    val pii = piiDetector(config.inputGuardrails.head)
    pii.piiTypes should contain theSameElementsAs Seq(PIIType.CreditCard, PIIType.SSN, PIIType.BankAccount)
    pii.onFail shouldBe GuardrailAction.Block
    injection(config.inputGuardrails(1)).sensitivity shouldBe InjectionSensitivity.High
    val t = topic(config.inputGuardrails(2))
    t.allowedTopics shouldBe topics
    t.threshold shouldBe 0.7
    t.onFail shouldBe GuardrailAction.Block

    config.outputGuardrails.map(_.getClass) shouldBe Seq(classOf[PIIMasker])
    masker(config.outputGuardrails.head).piiTypes should contain theSameElementsAs
      Seq(PIIType.CreditCard, PIIType.SSN, PIIType.BankAccount)

    config.ragGuardrails.map(_.getClass) shouldBe Seq(
      classOf[GroundingGuardrail],
      classOf[SourceAttributionGuardrail]
    )
    val g = grounding(config.ragGuardrails.head)
    g.threshold shouldBe 0.9
    g.strictMode shouldBe true
    attribution(config.ragGuardrails(1)).minAttributionScore shouldBe 0.8
    RAGGuardrails.allGuardrails(
      config
    ) should have size (config.inputGuardrails.size + config.outputGuardrails.size + config.ragGuardrails.size)
  }
}
