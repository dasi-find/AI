import json
import urllib.request


REQUEST_FILE = "sample_request_60.local.json"
API_URL = "http://127.0.0.1:8001/v1/rankings"


with open(
    REQUEST_FILE,
    "r",
    encoding="utf-8"
) as f:
    request_data = json.load(f)


body = json.dumps(
    request_data,
    ensure_ascii=False
).encode("utf-8")


request = urllib.request.Request(
    API_URL,
    data=body,
    headers={
        "Content-Type": "application/json; charset=utf-8"
    },
    method="POST"
)


with urllib.request.urlopen(request) as response:
    result = json.loads(
        response.read().decode("utf-8")
    )


print("=" * 60)
print("Smoke Test")
print("=" * 60)

print(
    "전체 Ranking:",
    result["totalRankedCount"]
)

print(
    "Top-K:",
    result["topK"]
)


print()
print("===== Top Candidates =====")

for candidate in result["topCandidates"]:

    print()
    print(
        f'{candidate["displayRank"]}위'
    )

    print(
        "itemId:",
        candidate["itemId"]
    )

    print(
        "finalScore:",
        candidate["finalScore"]
    )

    print(
        "reasons:"
    )

    for reason in candidate["reasons"]:
        print(
            " -",
            reason
        )


with open(
    "smoke_test_result.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )