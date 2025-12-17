import reflex as rx

config = rx.Config(
    app_name="llm_demo",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)