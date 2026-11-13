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

    topic = state["topic"]
    research_results = state["research_results"]
    sources = state["sources"]

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
            4. Use only the provided research.
            5. Do not invent facts or statistics.
            6. Do not generate or modify source URLs.

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

        summary = response.content

        # Add original sources received from Research Agent
        formatted_sources = "\n\nSources:\n"

        for index, url in enumerate(sources, start=1):
            formatted_sources += f"{index}. {url}\n"

        # Prepare final notification content
        final_summary = (
            f"Topic: {topic}\n\n"
            f"Key Highlights:\n{summary}"
            f"{formatted_sources}"
        )

        print("\nSummary generated successfully!")
        print(final_summary)

        return {
            "summary": final_summary,
            "sources": sources
        }

    except Exception as e:

        print(f"Summarization failed: {e}")

        return {
            "summary": "",
            "error": str(e)
        }
