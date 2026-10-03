from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEMO_RESPONSES: dict[str, dict[str, str]] = {
    "binary search": {
        "explanation": "Binary search finds a value in a sorted list by repeatedly checking the middle item. If the middle item is too small, the search continues in the upper half; if it is too large, it continues in the lower half. Each step removes about half of the remaining candidates, so the search takes O(log n) comparisons. The input must be sorted for this method to work.",
        "deep_dive": "Binary search maintains an interval with low and high indexes. Each iteration computes a midpoint, compares its value with the target, and discards one half of the interval. The invariant is that, if the target exists, it remains inside the interval. With n items the interval is halved at most about log2(n) times, giving O(log n) time and O(1) extra space for an iterative implementation. Use mid = low + (high - low) // 2 to avoid overflow in languages with fixed-width integers. Duplicate values require a defined policy: ordinary binary search may return any match, while lower-bound search finds the first matching position.",
        "simplified": "Binary search is a fast way to find something in an ordered list. Look at the middle. If your answer should be earlier, ignore the right half. If it should be later, ignore the left half. Repeat. Because half the list disappears each time, it is much faster than checking every item. The list must already be sorted.",
        "example": "Imagine finding page 72 in a 128-page book. Open near the middle, page 64. Since 72 is later, ignore pages 1–64. Check the middle of the remaining pages, around page 96. Since 72 is earlier, ignore 96–128. Continue halving the range until you reach page 72. A sorted list works the same way.",
    },
    "dijkstra's algorithm": {
        "explanation": "Dijkstra's algorithm finds shortest paths from one starting vertex to every other vertex in a weighted graph, as long as edge weights are non-negative. It keeps the best known distance to each vertex. Repeatedly, it selects the unvisited vertex with the smallest known distance and relaxes its outgoing edges. A priority queue makes this efficient, commonly O((V + E) log V) with an adjacency list and binary heap.",
        "deep_dive": "Dijkstra's algorithm initializes dist[source] = 0 and all other distances to infinity. A min-priority queue stores candidate (distance, vertex) pairs. When a vertex is removed, stale queue entries are skipped; otherwise each outgoing edge (u, v, w) is relaxed when dist[u] + w < dist[v]. The predecessor of v is updated at the same time, allowing the path to be reconstructed. With a binary heap and adjacency lists, the running time is O((V + E) log V). The greedy choice is correct because non-negative edges ensure a finalized minimum-distance vertex cannot later be improved. Negative edge weights break that reasoning; Bellman-Ford is appropriate when negative weights are allowed.",
        "simplified": "Dijkstra's algorithm finds the cheapest route through a network. Start at one place and write down the best cost you know for each nearby place. Always visit the not-yet-finished place with the lowest cost, then see whether going through it makes another route cheaper. It works when road costs are never negative.",
        "example": "Suppose A connects to B for 4 minutes and C for 1 minute; C connects to B for 2 minutes. Start at A: best costs are B=4 and C=1. Visit C first, then discover A→C→B costs 1+2=3, which improves B from 4 to 3. The shortest route to B is therefore through C.",
    },
    "dbms normalization": {
        "explanation": "Database normalization organizes tables to reduce duplicated data and prevent update problems. Instead of storing a customer's details in every order row, store customers once and refer to them by an ID. Normal forms such as 1NF, 2NF, and 3NF describe increasingly strong rules for table structure. Good normalization improves consistency, though some systems intentionally denormalize for performance.",
        "deep_dive": "Normalization uses functional dependencies to structure relational tables. First normal form (1NF) requires atomic values and no repeating groups. Second normal form (2NF) requires 1NF and removal of partial dependencies on part of a composite key. Third normal form (3NF) requires 2NF and removal of transitive dependencies: non-key attributes should depend on the key, the whole key, and nothing but the key. Decomposition should be lossless so joins reconstruct the original facts, and ideally preserve dependencies. Normalization reduces insertion, update, and deletion anomalies; denormalization may later be chosen deliberately to reduce read joins.",
        "simplified": "Normalization means splitting information into sensible tables so the same fact is not copied everywhere. Keep each kind of information in one place and connect tables using IDs. Then changing a customer's address only requires one update, instead of fixing many order rows.",
        "example": "Instead of a single Orders table repeating customer name and address on every purchase, make a Customers table with CustomerID, name, and address, plus an Orders table with OrderID and CustomerID. If the address changes, update one customer row. Orders still connect to the customer through CustomerID.",
    },
    "machine learning": {
        "explanation": "Machine learning is a way for a computer to learn patterns from examples and use them to make predictions. In supervised learning, examples include both inputs and correct labels. A model is trained on some examples and evaluated on separate data to estimate how well it generalizes. The quality of the data and evaluation matters as much as the choice of algorithm.",
        "deep_dive": "A supervised machine-learning pipeline represents examples as feature vectors x and targets y, then fits a model that approximates a mapping from x to y. Training minimizes a loss function, often with regularization to limit overfitting. A validation set supports model selection, while a held-out test set estimates final generalization. Data leakage occurs when information unavailable at prediction time enters training or evaluation. Useful measures depend on the task: accuracy can mislead on imbalanced classes, where precision, recall, F1, or a confusion matrix provide more detail. Feature extraction, preprocessing, and split strategy must be consistent between training and inference.",
        "simplified": "Machine learning is like learning from practice examples. You show a computer many examples and their answers. It looks for patterns, then tries to answer new examples it has not seen before. We test it on separate examples to check whether it learned a useful pattern instead of just memorizing.",
        "example": "To recognize hand gestures, collect hand landmark coordinates labeled Open Palm, Fist, or other gestures. Train a classifier on those labeled examples. When a new hand appears, extract the same landmarks and ask the model which learned pattern is the closest match.",
    },
    "operating system": {
        "explanation": "An operating system manages a computer's hardware and provides services to applications. It schedules programs on the CPU, manages memory and files, controls devices, and enforces permissions. Applications use operating-system interfaces instead of directly managing every hardware detail. Examples include Windows, macOS, Linux, Android, and iOS.",
        "deep_dive": "An operating system sits between applications and hardware. Its kernel handles privileged operations such as process scheduling, virtual memory, device drivers, file systems, and system calls. Processes receive isolated address spaces; virtual memory maps those spaces onto physical memory and can move less-used pages to storage. The scheduler allocates CPU time among runnable processes, while synchronization primitives coordinate shared resources. User mode limits application privileges, and kernel mode permits trusted OS code to manage hardware. These abstractions enable multiple applications to share a machine safely and efficiently.",
        "simplified": "An operating system is the manager of a computer. It decides which program gets to use the processor, keeps track of memory, organizes files, and helps programs communicate with devices such as the screen and keyboard. It lets people use applications without managing the hardware themselves.",
        "example": "When you open a music player and a browser at the same time, the operating system shares processor time between them, gives each program memory, reads music from storage, and sends sound to the speakers. The programs request these services rather than controlling the hardware directly.",
    },
}

ACTION_FIELD = {
    "EXPLAIN": "explanation",
    "DEEP DIVE": "deep_dive",
    "SIMPLIFY": "simplified",
    "EXAMPLE": "example",
}


class AIServiceError(RuntimeError):
    pass


class AIService:
    """OpenAI-compatible provider with a deterministic local study-demo fallback."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def mode(self) -> str:
        return "API configured" if self.api_key else "Demo Mode"

    @staticmethod
    def _topic_key(topic: str) -> str | None:
        normalized = topic.casefold().replace("’", "'")
        aliases = {
            "binary search": ("binary search",),
            "dijkstra's algorithm": ("dijkstra", "shortest path algorithm"),
            "dbms normalization": ("normalization", "normalisation", "dbms"),
            "machine learning": ("machine learning", "random forest", "supervised learning"),
            "operating system": ("operating system", "operating systems", "kernel"),
        }
        for key, terms in aliases.items():
            if any(term in normalized for term in terms):
                return key
        return None

    def _demo_response(self, action: str, topic: str, current_response: str) -> str:
        topic_key = self._topic_key(topic)
        field = ACTION_FIELD[action]
        if topic_key is not None:
            return DEMO_RESPONSES[topic_key][field]

        if action == "DEEP DIVE" and current_response:
            return (
                f"A deeper look at {topic}:\n\n{current_response}\n\n"
                "To analyze this further, identify the key components, how they interact, and the assumptions that affect the result."
            )
        if action == "SIMPLIFY" and current_response:
            return (
                f"In simpler terms, {topic} is about understanding one main idea and how its parts work together.\n\n"
                f"{current_response}"
            )
        if action == "EXAMPLE":
            return (
                f"A practical way to explore {topic} is to choose a small, familiar case, work through it one step at a time, "
                "and check how each decision changes the result."
            )
        return (
            f"{topic} is a useful study topic. Start by identifying its purpose, the main concepts, and how those concepts "
            "connect. In Demo Mode, GestureAI includes prepared study notes for Binary Search, Dijkstra's Algorithm, "
            "DBMS Normalization, Machine Learning, and Operating Systems."
        )

    def _request(self, action: str, topic: str, current_response: str) -> str:
        instructions = {
            "EXPLAIN": "Explain the topic clearly for a student. Be accurate and concise, with a short example when useful.",
            "DEEP DIVE": "Expand the existing explanation in detail. Stay on the same topic and preserve its meaning. Include key concepts, examples, and technical detail where useful.",
            "SIMPLIFY": "Rewrite the current explanation in simpler student-friendly language. Preserve its meaning and use an analogy if helpful.",
            "EXAMPLE": "Give a practical example for the same topic. Include a simple analogy and a small worked example when useful.",
        }
        user_content = f"Topic/question: {topic}"
        if current_response and action in {"DEEP DIVE", "SIMPLIFY"}:
            user_content += f"\n\nCurrent explanation to transform:\n{current_response}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions[action]},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.4,
            "max_tokens": 700,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=25) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()
            if not content:
                raise AIServiceError("AI service returned an empty response.")
            return content
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AIServiceError("AI service is temporarily unavailable.") from exc

    def generate(self, action: str, topic: str, current_response: str = "") -> str:
        if action not in ACTION_FIELD:
            raise ValueError(f"Unsupported study action: {action}")
        if self.api_key:
            return self._request(action, topic, current_response)
        return self._demo_response(action, topic, current_response)
