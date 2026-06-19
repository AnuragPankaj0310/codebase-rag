from storage import show_results

queries = [

    "Flask application initialization",

    "request routing",

    "celery task implementation",

    "how does flask create app",

    "url route decorators"
]

for query in queries:

    show_results(
        query,
        top_k=3
    )