import asyncio
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
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str
    question: Optional[str]


class SummaryState(TypedDict):
    content: str
    video_title: str


class QuestionState(TypedDict):
    content: str
    video_title: str
    question: str


class Summarizer:
    SECTION_TITLES_PROMPT = """Your mission is to summarize a video using its title and English subtitles. 
    Your goal is to formulate the insightful notes a careful viewer would make, not to create a detailed, verbatim recital. Prioritize brevity over exhaustiveness.
    The subtitles are formatted as [timestamp in seconds]: [subtitle].
    For each sentence in your summary, provide the timestamp corresponding to the relevant part of the video. For example, a summary sentence based on content starting at 31 seconds should be formatted as [31]: Summary sentence..
    
    The video title is: {video_title}
    The subtitles are provided between the triple backticks:
    ```
    {text}
    ```

    Your summary:
    """
    
    SUMMARY_PROMPT = """Your mission is to write a concise summary of a video using its title and chapter summaries.
    The chapter summaries are formatted as [timestamp in seconds]: chapter summary.
    The video title is: {video_title}
    The chapter summaries are provided between the triple backticks:
    ```
    {text}
    ```

    Your concise video summary:"""
    
    QUESTION_PROMPT = """Your mission is to answer a question about a video using its title and English subtitles. Answer the question directly and concisely based only on the information within the provided subtitles. If the answer cannot be found, state that clearly.
    The subtitles are formatted as [timestamp in seconds]: [subtitle].
    For each sentence in your answer, provide the timestamp corresponding to the relevant part of the subtitles. For example, an answer sentence based on content starting at 31 seconds should be formatted as [31]: Answer sentence..
    
    The question is: {question}
    The video title is: {video_title}
    The subtitles are provided between the triple backticks:
    ```
    {text}
    ```

    Your answer:
    """
    
    SECTION_PROMPT_TEMPLATE = PromptTemplate(
        template=SECTION_TITLES_PROMPT, input_variables=["text", "video_title"]
    )
    
    OVERALL_PROMPT_TEMPLATE = PromptTemplate(
        template=SUMMARY_PROMPT, input_variables=["text", "video_title"]
    )
    
    QUESTION_PROMPT_TEMPLATE = PromptTemplate(
        template=QUESTION_PROMPT, input_variables=["text", "video_title", "question"]
    )

    def __init__(self, max_summary_len_tokens: int = 500, chunk_size: int = 8000):
        self.max_summary_len_tokens = max_summary_len_tokens
        self.chunk_size = chunk_size
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini", max_completion_tokens=self.max_summary_len_tokens)
        
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=0,
            separator="\n"
        )
        
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OverallState)
        
        graph.add_node("generate_summary", self._generate_section_summary)
        graph.add_node("generate_question_answer", self._generate_section_question_answer)
        graph.add_node("collect_summaries", self._collect_summaries)
        graph.add_node("generate_final_summary", self._generate_final_summary)
        
        graph.add_conditional_edges(START, self._map_initial, ["generate_summary", "generate_question_answer"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_edge("generate_question_answer", "collect_summaries")
        graph.add_edge("collect_summaries", "generate_final_summary")
        graph.add_edge("generate_final_summary", END)
        
        return graph.compile()

    def _length_function(self, documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)
    
    def _map_initial(self, state: OverallState):
        """Map each content chunk to either summary or question node."""
        if state.get("question"):
            return [
                Send("generate_question_answer", {
                    "content": content, 
                    "video_title": state["video_title"],
                    "question": state["question"]
                }) 
                for content in state["contents"]
            ]
        else:
            return [
                Send("generate_summary", {"content": content, "video_title": state["video_title"]}) 
                for content in state["contents"]
            ]

    async def _generate_section_summary(self, state: SummaryState):
        """Generate summary for a section."""
        prompt = self.SECTION_PROMPT_TEMPLATE.invoke({
            "text": state["content"], 
            "video_title": state["video_title"]
        })
        response = await self.llm.ainvoke(prompt)
        return {"summaries": [response.content]}

    async def _generate_section_question_answer(self, state: QuestionState):
        """Generate question answer for a section."""
        prompt = self.QUESTION_PROMPT_TEMPLATE.invoke({
            "text": state["content"], 
            "video_title": state["video_title"],
            "question": state["question"]
        })
        response = await self.llm.ainvoke(prompt)
        return {"summaries": [response.content]}

    def _collect_summaries(self, state: OverallState):
        """Collect all summaries into documents."""
        return {
            "collapsed_summaries": [Document(page_content=summary) for summary in state["summaries"]]
        }

    async def _reduce(self, documents: List[Document], video_title: str, question: Optional[str] = None) -> str:
        """Reduce documents to a single summary or answer."""
        combined_text = "\n".join([doc.page_content for doc in documents])
        
        if question:
            prompt = self.QUESTION_PROMPT_TEMPLATE.invoke({
                "text": combined_text,
                "video_title": video_title,
                "question": question
            })
        else:
            prompt = self.OVERALL_PROMPT_TEMPLATE.invoke({
                "text": combined_text,
                "video_title": video_title
            })
        
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def _generate_final_summary(self, state: OverallState):
        """Generate the final summary, collapsing if necessary."""
        documents = state["collapsed_summaries"]
        video_title = state["video_title"]
        question = state.get("question")
        final_summary = await self._reduce(documents, video_title, question)
        return {"final_summary": final_summary}

    def summarize(self, video_title: str, subtitles: str, question: Optional[str] = None) -> tuple[List[SectionSummary], str]:
        """
        Summarize video subtitles into sections and overall summary, or answer a question.
        
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
            "summaries": [],
            "collapsed_summaries": [],
            "final_summary": "",
            "question": question
        }))
        
        section_summaries = []
        for summary_text in result.get("summaries", []):
            section_summaries.extend(self._parse_section_summaries_text(summary_text))
        
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
