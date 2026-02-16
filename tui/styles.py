APP_CSS = """
#main {
    height: 1fr;
}
#chat-panel {
    width: 1fr;
    min-width: 30;
    border: solid $primary;
    padding: 0 1;
}
#chat-title, #inspector-title {
    text-style: bold;
    padding: 0 1;
    background: $surface;
    height: 3;
    content-align: center middle;
}
#inspector-panel {
    border: solid $secondary;
    padding: 0 1;
}
#inspector-tabs {
    height: 2fr;
    min-height: 18;
}
#chat-log {
    height: 1fr;
}
#inspector-log {
    height: 2fr;
}
#bottom-info {
    height: 1fr;
    min-height: 12;
    padding-top: 1;
}
#context-view-panel {
    height: 1fr;
    min-height: 8;
    margin-bottom: 1;
}
#token-panel {
    height: auto;
    max-height: 16;
    padding: 1 1;
    border: solid $accent;
    margin-bottom: 1;
}
#concept-panel {
    height: auto;
    max-height: 5;
    padding: 0 1;
    margin-bottom: 1;
}
#docs-label {
    height: auto;
    max-height: 3;
    color: $text-muted;
    padding: 0 1;
}
#chat-input {
    dock: bottom;
}
#slash-popup {
    height: 7;
    border: solid $accent;
    padding: 0 1;
    margin: 1 0 0 0;
    overflow-y: auto;
    display: none;
}
"""
