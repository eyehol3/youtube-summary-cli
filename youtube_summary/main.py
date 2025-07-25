from typing import List, Optional
import typer
import shutil
from langchain_community.callbacks import get_openai_callback
from rich.console import Console
from rich.panel import Panel
from rich.padding import Padding
from typer.rich_utils import (
    ALIGN_ERRORS_PANEL,
    ERRORS_PANEL_TITLE,
    STYLE_ERRORS_PANEL_BORDER,
)
from youtube_summary.summarizer import Summarizer, SectionSummary
from youtube_summary.transcript import get_transcripts
from youtube_summary.video_infromation import extract_video_information

app = typer.Typer()


def pretty_timestamp(timestamp_seconds: int) -> str:
    hours, remainder = divmod(timestamp_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_pretty_section_summary_text(
    url: str, section_summaries: List[SectionSummary], text_max_width: int
) -> str:
    pretty_summaries = []
    for section_summary in section_summaries:
        timestamp_seconds = section_summary.timestamp_seconds
        link = f"{url}&t={timestamp_seconds}"
        timestamp_pretty = pretty_timestamp(timestamp_seconds)
        summary = f"• [link={link}]{timestamp_pretty}[/link]: {section_summary.text}"
        pretty_summaries.append(summary)
    return "\n\n".join(pretty_summaries)


def pretty_print_exception_message(console: Console, e: Exception) -> None:
    console.print(
        Panel(
            f"An error occurred: {e}",
            border_style=STYLE_ERRORS_PANEL_BORDER,
            title=ERRORS_PANEL_TITLE,
            title_align=ALIGN_ERRORS_PANEL,
            highlight=False,
        )
    )


@app.command()
def main(
    url: str, 
    debug_mode: bool = False,
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Ask a specific question about the video instead of generating a summary")
):
    """
    A simple CLI tool that summarizes YouTube videos or answers questions about them.

    If you encounter a bug, please open an issue at: https://github.com/mmaorc/youtube-summary-cli.

    If you have any questions, you can find me on Twitter: https://bit.ly/43ow5WT.
    """
    text_max_width = int(shutil.get_terminal_size().columns*0.7)
    padding = 2
    console = Console(highlight=False, width=text_max_width)
    err_console = Console(stderr=True)

    try:
        summarizer = Summarizer()

        video_information = extract_video_information(url)
        subtitles = get_transcripts(video_information.id)

        with get_openai_callback() as cb:
            section_summaries, output = summarizer.summarize(
                video_information.title, subtitles, question
            )

        if debug_mode:
            console.print(f"[dim]Debug - Section summaries count: {len(section_summaries)}[/dim]")
            console.print(f"[dim]Debug - Output length: {len(output)}[/dim]")
            if section_summaries:
                console.print(f"[dim]Debug - First section summary: {section_summaries[0]}[/dim]")

        console.print()
        console.print(
            f"[bold]Title:[/bold] [link={video_information.url}]"
            f"{video_information.title}[/link]"
        )
        video_duration = pretty_timestamp(video_information.duration)
        console.print(f"[bold]Duration:[/bold] {video_duration}")
        console.print(
            f"[bold]Channel:[/bold] [link={video_information.channel_url}]"
            f"{video_information.channel}[/link]"
        )

        console.print()
        if question:
            console.print(f"[bold]Question:[/bold] {question}")
            console.print()
            console.print("[bold]Answer:[/bold]")
            if not section_summaries:
                console.print(Padding(output, (0, padding)))
            else:
                console.print(Padding(get_pretty_section_summary_text(url, section_summaries, text_max_width), (0, padding)))
        
        else:
            console.print("[bold]Summary:[/bold]")
            console.print(Padding(output, (0, padding)))

            console.print()
            console.print("[bold]Chapter Summaries:[/bold]")
            console.print(Padding(get_pretty_section_summary_text(url, section_summaries, text_max_width), (0, padding)))
            if not section_summaries:
                console.print("[dim]No timestamped sections found.[/dim]")

        console.print()
        console.print("[bold]OpenAI Stats:[/bold]")
        console.print(cb)
    except Exception as e:
        if debug_mode:
            raise e
        pretty_print_exception_message(err_console, e)
        raise typer.Exit(1)


if __name__ == "__main__":
    url='https://www.youtube.com/watch?v=XbLDeWYBZw4'

    section_summaries=[SectionSummary(timestamp_seconds=4, text='At least four trillion dollars per year by 2030 is urgently needed to avert a climate crisis, yet this capital is not flowing fast enough despite awareness and excitement among over 450 financial institutions managing $130 trillion committed to net-zero by 2050.  '), SectionSummary(timestamp_seconds=71, text='Deploying real money into early-stage green technologies and affordable, low-emission housing is difficult due to risk thresholds and insufficient demand, leaving much capital idle.  '), SectionSummary(timestamp_seconds=110, text='Housing affordability, energy inefficiency, and racial equity are deeply linked, with over seven million affordable homes lacking for low-income renters, disproportionately affecting Black and Brown communities who pay higher utility costs.  '), SectionSummary(timestamp_seconds=193, text='Buildings significantly contribute to greenhouse gas emissions, and business-as-usual construction will worsen climate change, highlighting the need for sustainable finance to transform these challenges into investment opportunities.  '), SectionSummary(timestamp_seconds=239, text='Banks can finance over a trillion dollars in emissions-reducing housing solutions, but early-stage materials and low-return projects in disadvantaged areas require public policies, capital, and demand to meet risk and return thresholds.  '), SectionSummary(timestamp_seconds=308, text='Mobilizing trillions requires making the climate-housing link a government priority, with blended finance—government guarantees and concessional lending—de-risking private investment and proving effective in affordable housing.  '), SectionSummary(timestamp_seconds=377, text='Governments must enact the "three p\'s": public policies, programs, and permits; lack of energy-efficiency standards, insufficient homeowner incentives, and permitting delays stall financing despite banks’ net-zero ambitions.  '), SectionSummary(timestamp_seconds=441, text='Voters can pressure local governments to implement these policies and programs, where small electorates can drive significant climate finance action by enabling banks to deploy capital locally.  '), SectionSummary(timestamp_seconds=504, text='Institutional investors like pension plans and insurance companies manage vast long-term funds and increasingly recognize climate and social risks, having financed $140 billion in social bonds in 2020, but scaling to trillions requires mandates integrating environmental and social criteria.  '), SectionSummary(timestamp_seconds=588, text='Investors offer green investment options, and public demand for such choices strengthens mandates to allocate capital toward sustainable projects; consumers should request and support these options.  '), SectionSummary(timestamp_seconds=633, text='Individuals can create demand for energy-efficient, low-carbon homes by prioritizing these features when buying or renting, encouraging banks and developers to finance greener housing akin to shifts seen in food and transportation sectors.  '), SectionSummary(timestamp_seconds=701, text='The three key actions are activating governments, mobilizing institutional investors, and generating consumer demand to unlock trillions for climate solutions, recognizing that money influences how long humanity can thrive on this planet.')]
    section_summaries=[SectionSummary(timestamp_seconds=4, text='At least four trillion dollars per year by 2030 is urgently needed to avert a climate crisis, yet this capital is not flowing fast enough despite awareness and excitement among over 450 financial institutions managing $130 trillion committed to net-zero by 2050.  '), SectionSummary(timestamp_seconds=71, text='Deploying real money into early-stage green technologies and affordable, low-emission housing is difficult due to risk thresholds and insufficient demand, leaving much capital idle.  '), SectionSummary(timestamp_seconds=110, text='Housing affordability, energy inefficiency, and racial equity are deeply linked, with over seven million affordable homes lacking for low-income renters, disproportionately affecting Black and Brown communities who pay higher utility costs.  '), SectionSummary(timestamp_seconds=193, text='Buildings significantly contribute to greenhouse gas emissions, and business-as-usual construction will worsen climate change, highlighting the need for sustainable finance to transform these challenges into investment opportunities.  '), SectionSummary(timestamp_seconds=239, text='Banks can finance over a trillion dollars in emissions-reducing housing solutions, but early-stage materials and low-return projects in disadvantaged areas require public policies, capital, and demand to meet risk and return thresholds.  '), SectionSummary(timestamp_seconds=308, text='Mobilizing trillions requires making the climate-housing link a government priority, with blended finance—government guarantees and concessional lending—de-risking private investment and proving effective in affordable housing.  '), SectionSummary(timestamp_seconds=377, text='Governments must enact the "three p\'s": public policies, programs, and permits; lack of energy-efficiency standards, insufficient homeowner incentives, and permitting delays stall financing despite banks’ net-zero ambitions.  '), SectionSummary(timestamp_seconds=441, text='Voters can pressure local governments to implement these policies and programs, where small electorates can drive significant climate finance action by enabling banks to deploy capital locally.  '), SectionSummary(timestamp_seconds=504, text='Institutional investors like pension plans and insurance companies manage vast long-term funds and increasingly recognize climate and social risks, having financed $140 billion in social bonds in 2020, but scaling to trillions requires mandates integrating environmental and social criteria.  '), SectionSummary(timestamp_seconds=588, text='Investors offer green investment options, and public demand for such choices strengthens mandates to allocate capital toward sustainable projects; consumers should request and support these options.  '), SectionSummary(timestamp_seconds=633, text='Individuals can create demand for energy-efficient, low-carbon homes by prioritizing these features when buying or renting, encouraging banks and developers to finance greener housing akin to shifts seen in food and transportation sectors.  '), SectionSummary(timestamp_seconds=701, text='The three key actions are activating governments, mobilizing institutional investors, and generating consumer demand to unlock trillions for climate solutions, recognizing that money influences how long humanity can thrive on this planet.')]
    text_max_width=86
    console = Console(highlight=False, width=text_max_width)
    console.print(Padding(get_pretty_section_summary_text(url, section_summaries, text_max_width), (0,2)))