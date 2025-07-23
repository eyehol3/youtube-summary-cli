import asyncio
import json
import operator
import re
from dataclasses import dataclass
from typing import Annotated, List, TypedDict, Optional

from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph


@dataclass(frozen=True)
class SectionSummary:
    timestamp_seconds: int
    text: str


class OverallState(TypedDict):
    video_title: str
    contents: List[str]
    dense_summaries: Annotated[list, operator.add]
    final_summary: str
    section_responses: Annotated[list, operator.add]
    question: Optional[str]
    relevant_section_ids: Optional[List[int]]


class DenseSummaryState(TypedDict):
    content: str
    video_title: str
    section_id: int


class SectionRefinementState(TypedDict):
    content: str
    dense_summary: str
    video_title: str


class QuestionGradingState(TypedDict):
    dense_summaries: List[str]
    question: str


class QuestionAnswerState(TypedDict):
    combined_content: str
    video_title: str
    question: str


class Summarizer:
    DENSE_SUMMARY_PROMPT = """Video Content: {text}

You will generate increasingly concise, entity-dense summaries of the above video content. 
Repeat the following 2 steps 5 times. 
Step 1. Identify 1-3 informative entities (";" delimited) from the video content which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 

A missing entity is:
- relevant to the main story, 
- specific yet concise (5 words or fewer), 
- novel (not in the previous summary), 
- faithful (present in the video content), 
- anywhere (can be located anywhere in the video content).

Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this video discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the video discusses".
- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the video content. 
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
Remember, use the exact same number of words for each summary.

The video title is: {video_title}

Answer in JSON (without backticks). The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""

    SECTION_REFINEMENT_PROMPT = """Your mission is to synthesize a dense, timestamped summary. You will use a high-level dense summary to guide the compression of the original, timestamped video content.

Dense Summary:
{dense_summary}

Video Content (with timestamps):
{content}

The video title is: {video_title}

Instructions:
- Identify Core Concepts: Use the Dense Summary as a guide to identify the most important topics and entities within the Video Content.
- Synthesize and Compress: Combine multiple related points from the Video Content into single, denser summary sentences. Do not create a one-to-one mapping from the original subtitles. The goal is to reduce verbosity and increase information density.
- Eliminate Filler: Make every word count. Aggressively remove uninformative phrases and filler language (e.g., "this video talks about," "as the speaker mentions").
- Preserve Origin Timestamps: Each synthesized sentence in your output must begin with the timestamp of the *earliest* piece of original content it references.
- Follow Format: The final output must be a list of timestamped sentences in the format `[timestamp]: summary sentence`.

Your refined timestamped summary:"""

    QUESTION_GRADING_PROMPT = """Evaluate which video sections are relevant to answer the given question.

Question: {question}

Video Section Summaries:
{dense_summaries}

For each section (numbered 0 to {num_sections}), respond with "relevant" or "not_relevant".
Return only a JSON list of strings, one for each section in order.
Example: ["relevant", "not_relevant", "relevant"]"""

    QUESTION_ANSWER_PROMPT = """Your mission is to synthesize a direct and concise answer to the question using the provided video content.

Question: {question}
Video Title: {video_title}

Original Video Content (with timestamps):
{original_content}

Dense Summaries of Relevant Sections:
{dense_summaries}

Instructions:
- Synthesize a Direct Answer: Formulate a comprehensive answer to the question. Do not simply list facts chronologically from the video. Your goal is to synthesize information from across the video into a cohesive, direct response.
- Use Sources Strategically: The Original Video Content is your source of truth for facts and timestamps. The Dense Summaries are your guide for identifying the most important concepts to include in your answer.
- Compress and Fuse: Combine multiple related points from the Original Video Content into single, denser sentences. A single sentence in your answer can draw from several different timestamps.
- Timestamp Correctly: Each sentence in your answer must begin with the timestamp of the *earliest* piece of original content it references.
- Be Concise: Aggressively remove filler language. Every word must contribute directly to the answer.
- Format: Your final answer must be a series of timestamped sentences in the format `[timestamp]: Answer sentence.`

Your answer::"""

    DENSE_SUMMARY_TEMPLATE = PromptTemplate(
        template=DENSE_SUMMARY_PROMPT, input_variables=["text", "video_title"]
    )
    
    SECTION_REFINEMENT_TEMPLATE = PromptTemplate(
        template=SECTION_REFINEMENT_PROMPT, input_variables=["dense_summary", "content", "video_title"]
    )
    
    QUESTION_GRADING_TEMPLATE = PromptTemplate(
        template=QUESTION_GRADING_PROMPT, input_variables=["question", "dense_summaries", "num_sections"]
    )
    
    QUESTION_ANSWER_TEMPLATE = PromptTemplate(
        template=QUESTION_ANSWER_PROMPT, input_variables=["question", "video_title", "dense_summaries", "original_content"]
    )

    def __init__(self, max_summary_len_tokens: int = 1000, chunk_size: int = 8000):
        self.max_summary_len_tokens = max_summary_len_tokens
        self.chunk_size = chunk_size
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini", max_completion_tokens=self.max_summary_len_tokens)
        
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=200,  # Small overlap as requested
            separator="\n"
        )
        
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OverallState)
        
        graph.add_node("generate_dense_summaries", self._generate_dense_summary)
        graph.add_node("generate_overall_summary", self._generate_overall_summary)
        graph.add_node("refine_sections", self._refine_section)
        graph.add_node("grade_sections", self._grade_sections_for_question)
        graph.add_node("answer_question", self._answer_question)
        
        graph.add_conditional_edges(START, self._map_to_dense_summaries, ["generate_dense_summaries"])
        graph.add_conditional_edges("generate_dense_summaries", self._route_after_dense_summaries, 
                                   ["generate_overall_summary", "grade_sections"])
        graph.add_conditional_edges("generate_overall_summary", self._map_to_section_refinement, ["refine_sections"])
        graph.add_edge("refine_sections", END)
        graph.add_edge("grade_sections", "answer_question")
        graph.add_edge("answer_question", END)
        
        return graph.compile()

    def _map_to_dense_summaries(self, state: OverallState):
        """Map each content chunk to dense summary generation."""
        return [
            Send("generate_dense_summaries", {
                "content": content, 
                "video_title": state["video_title"],
                "section_id": i
            }) 
            for i, content in enumerate(state["contents"])
        ]

    def _route_after_dense_summaries(self, state: OverallState):
        """Route to either overall summary generation or question grading."""
        if state.get("question"):
            return "grade_sections"
        else:
            return "generate_overall_summary"

    def _map_to_section_refinement(self, state: OverallState):
        """Map each section to refinement."""
        return [
            Send("refine_sections", {
                "content": state["contents"][i],
                "dense_summary": dense_summary,
                "video_title": state["video_title"]
            })
            for i, dense_summary in enumerate(state["dense_summaries"])
        ]

    async def _generate_dense_summary(self, state: DenseSummaryState):
        """Generate dense summary for a section."""
        prompt = self.DENSE_SUMMARY_TEMPLATE.invoke({
            "text": state["content"], 
            "video_title": state["video_title"]
        })
        response = await self.llm.ainvoke(prompt)
        
        try:
            # Parse JSON response and extract final dense summary
            summaries = json.loads(response.content)
            final_dense_summary = summaries[-1]["Denser_Summary"]  # Use the 5th iteration
        except (json.JSONDecodeError, KeyError, IndexError):
            # Fallback to raw response if JSON parsing fails
            final_dense_summary = response.content
            
        return {"dense_summaries": [final_dense_summary]}

    async def _generate_overall_summary(self, state: OverallState):
        """Generate overall summary from dense summaries."""
        if len(state["dense_summaries"]) == 1:
            # Single chunk, check if length exceeds max_summary_len
            if self.llm.get_num_tokens(state["dense_summaries"][0]) <= self.max_summary_len_tokens:
                return {"final_summary": state["dense_summaries"][0]}
        
        # Multiple chunks or single long chunk - create dense summary of dense summaries
        combined_dense_summaries = "\n".join(state["dense_summaries"])
        prompt = self.DENSE_SUMMARY_TEMPLATE.invoke({
            "text": combined_dense_summaries,
            "video_title": state["video_title"]
        })
        response = await self.llm.ainvoke(prompt)
        
        try:
            summaries = json.loads(response.content)
            final_summary = summaries[-1]["Denser_Summary"]
        except (json.JSONDecodeError, KeyError, IndexError):
            final_summary = response.content
            
        return {"final_summary": final_summary}

    async def _refine_section(self, state: SectionRefinementState):
        """Refine section summary using dense summary insights."""
        prompt = self.SECTION_REFINEMENT_TEMPLATE.invoke({
            "dense_summary": state["dense_summary"],
            "content": state["content"],
            "video_title": state["video_title"]
        })
        response = await self.llm.ainvoke(prompt)
        return {"section_responses": [response.content]}

    async def _grade_sections_for_question(self, state: OverallState):
        """Grade which sections are relevant for the question."""
        dense_summaries_text = "\n".join([f"Section {i}: {summary}" for i, summary in enumerate(state["dense_summaries"])])
        
        prompt = self.QUESTION_GRADING_TEMPLATE.invoke({
            "question": state["question"],
            "dense_summaries": dense_summaries_text,
            "num_sections": len(state["dense_summaries"]) - 1
        })
        response = await self.llm.ainvoke(prompt)
        
        try:
            relevance_scores = json.loads(response.content)
            relevant_ids = [i for i, score in enumerate(relevance_scores) if score == "relevant"]
        except (json.JSONDecodeError, IndexError):
            # Fallback: use all sections
            relevant_ids = list(range(len(state["dense_summaries"])))
            
        return {"relevant_section_ids": relevant_ids}

    async def _answer_question(self, state: OverallState):
        """Answer question using relevant sections."""
        relevant_ids = state.get("relevant_section_ids", list(range(len(state["dense_summaries"]))))
        
        # Separate dense summaries and original content
        relevant_dense_summaries = []
        relevant_original_content = []
        
        for i in relevant_ids:
            relevant_dense_summaries.append(f"Section {i}: {state['dense_summaries'][i]}")
            relevant_original_content.append(f"Section {i}:\n{state['contents'][i]}")
        
        dense_summaries_text = "\n\n".join(relevant_dense_summaries)
        original_content_text = "\n\n".join(relevant_original_content)
        
        prompt = self.QUESTION_ANSWER_TEMPLATE.invoke({
            "question": state["question"],
            "video_title": state["video_title"],
            "dense_summaries": dense_summaries_text,
            "original_content": original_content_text
        })
        response = await self.llm.ainvoke(prompt)
        return {"section_responses": [response.content], "final_summary": response.content}

    def summarize(self, video_title: str, subtitles: str, question: Optional[str] = None) -> tuple[List[SectionSummary], str]:
        """
        Summarize video subtitles using chain-of-density approach, or answer a question.
        
        Args:
            video_title: Title of the video
            subtitles: Video subtitles text
            question: Optional question to answer instead of summarizing
        
        Returns:
            tuple: (section_summaries, overall_summary_or_answer)
        """
        docs = [Document(page_content=subtitles)]
        split_docs = self.text_splitter.split_documents(docs)
        chunks = [doc.page_content for doc in split_docs]
        
        result = asyncio.run(self.app.ainvoke({
            "video_title": video_title,
            "contents": chunks,
            "dense_summaries": [],
            "final_summary": "",
            "section_responses": [],
            "question": question,
            "relevant_section_ids": None
        }))
        
        section_summaries = []
        for response_text in result.get("section_responses", []):
            section_summaries.extend(self._parse_section_summaries_text(response_text))

        return section_summaries, result["final_summary"]

    @staticmethod
    def _parse_section_summaries_text(text: str) -> List[SectionSummary]:
        """Parse section summaries from text."""
        lines = text.split("\n")
        parsed_lines = []

        for line in lines:
            match = re.match(r"\s*\[(\d+(?:\.\d+)?)\]:\s+(.*)", line)
            if match:
                timestamp_seconds, text = match.groups()
                parsed_line = SectionSummary(
                    timestamp_seconds=int(float(timestamp_seconds)), text=text
                )
                parsed_lines.append(parsed_line)

        return parsed_lines
