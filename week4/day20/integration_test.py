"""
Day 20: Integration Testing
--------------------------------
Runs a batch of MIXED questions through the Day 19 router automatically —
some clearly document questions, some clearly tool questions, some plain
conversation, and one deliberately ambiguous one — so you can see all the
routing decisions at once and spot any mis-routes, instead of typing each
one by hand.

This imports router.py from day19 directly (reuses everything, no new
tools/retrieval code — Day 20 is testing, not building).
"""

import sys
sys.path.append("../day19")

from router import classify_question, handle_document, handle_tool, handle_conversation

# a mix of question types, plus one genuinely ambiguous one at the end
TEST_QUESTIONS = [
    ("how many words are in naila's daughter's name?", "???"),
    ("what year is it?", "conversation"),
    ("why?", "???"),
    ("what's 12 divided by 4?", "tool"),
    ("who found the wooden box?", "document"),
    ("count the words in naila's husband's name", "???"),
    ("what's your favorite color?", "conversation"),
    ("multiply the number of weeks by 5", "???"),
    ("is there a dog in the story?", "document"),
    ("what's 100 minus 37?", "tool"),
]


def run_integration_test():
    print("Day 20 Integration Test — running mixed questions through the router\n")
    print(f"{'Question':<45}{'Expected':<15}{'Got':<15}{'Match?'}")
    print("-" * 90)

    mismatches = []

    for question, expected in TEST_QUESTIONS:
        route = classify_question(question)
        match = "✓" if route == expected else ("?" if expected == "???" else "✗ MIS-ROUTE")

        print(f"{question:<45}{expected:<15}{route:<15}{match}")

        if match.startswith("✗"):
            mismatches.append((question, expected, route))

    print("\n--- Answers (so you can sanity-check the routing made sense) ---\n")

    for question, expected in TEST_QUESTIONS:
        route = classify_question(question)
        if route == "document":
            answer = handle_document(question)
        elif route == "tool":
            answer = handle_tool(question)
        else:
            answer = handle_conversation(question)

        print(f"You: {question}")
        print(f"[router -> {route}]")
        print(f"Bot: {answer}\n")

    if mismatches:
        print(f"\n{len(mismatches)} mis-route(s) found:")
        for q, expected, got in mismatches:
            print(f"  - '{q}' expected '{expected}' but got '{got}'")
    else:
        print("\nNo mis-routes on the clear-cut questions.")

    last_question = TEST_QUESTIONS[-1][0]
    last_expected = TEST_QUESTIONS[-1][1]
    if last_expected == "???":
        print(f"\nThe last question ('{last_question}') was marked as ambiguous — "
              "check above which path it took and whether the answer still made sense.")


if __name__ == "__main__":
    run_integration_test()