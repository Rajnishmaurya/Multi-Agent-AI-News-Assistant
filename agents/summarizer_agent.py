
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize OpenRouter LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


def summarizer_agent(state):

    print("\n--- SUMMARIZER AGENT STARTED ---")

    topic = state.get("topic", "")
    research_results = state.get("research_results", "")
    sources = state.get("sources", [])

    try:

        if not research_results:
            raise ValueError("No research findings received.")

        # Summarization prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI News Summarization Agent.

            Your task is to summarize research findings
            into exactly five key points.

            Instructions:
            1. Generate exactly five key points.
            2. Keep each point concise and informative.
            3. Focus on important technology developments.
            4. Use only the information provided in the research.
            5. Do not invent facts or statistics.
            6. Do not generate or modify source URLs.
            7. Return each key point on a separate line.
            8. Number the points from 1 to 5.

            Return only the five key points.
            """),

            ("human", """
            Topic: {topic}

            Research Findings:
            {research_results}
            """)
        ])

        # Execute LLM chain
        chain = prompt | llm

        response = chain.invoke({
            "topic": topic,
            "research_results": research_results
        })

        summary = response.content.strip()

        # Convert the response into five individual points
        summary_points = [
            line.strip()
            for line in summary.splitlines()
            if line.strip()
        ]

        # Remove numbering and bullet symbols
        summary_points = [
            point.lstrip("0123456789.-•) ").strip()
            for point in summary_points
        ]

        # Remove empty lines
        summary_points = [
            point for point in summary_points if point
        ]

        # Validate exactly five points
        if len(summary_points) != 5:
            raise ValueError(
                f"Expected exactly 5 summary points, "
                f"but received {len(summary_points)}."
            )

        print("\nSummary generated successfully!")

        for index, point in enumerate(summary_points, start=1):
            print(f"{index}. {point}")

        print(f"\nSources received: {len(sources)}")

        # Return data to the shared LangGraph state
        return {
            "summary_points": summary_points,
            "sources": sources
        }

    except Exception as e:

        print(f"Summarization failed: {e}")

        return {
            "summary_points": [],
            "sources": sources,
            "error": str(e)
        }
