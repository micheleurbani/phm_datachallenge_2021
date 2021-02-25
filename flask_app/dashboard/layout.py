html_layout = """
<!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
        </head>
        <body class="dash-template">
            <nav class="navbar expand-lg navbar-light bg-light">
                <div class="container fluid">
                    <a class="navbar-brand" href="/">PHM Challenge 2021</a>
                </div>
            </nav>
            <div class="container">
            {%app_entry%}
            </div>
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
"""
