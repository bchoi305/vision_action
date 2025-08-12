# PDF Collector

This application automates the process of downloading PDF articles from a given DOI.

## How it works

This application uses the Vision Action Agent to navigate websites, log in, and click download links.
It takes a DOI as input and follows a predefined workflow in the `config.yaml` file to download the corresponding PDF.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r ../vision_action_agent/requirements.txt
    ```

2.  **Configure your workflow:**
    Open `config.yaml` and replace the placeholder values with the actual text and URLs from the website you want to automate.
    You will need to provide:
    *   Your library or publisher's login URL.
    *   The text labels for the username, password, and DOI input fields.
    *   The text on the login and submit buttons.
    *   The text of the PDF download link.

## Usage

Run the application from the `pdfcollect` directory:

```bash
python main.py <DOI> \
  [--downloads-path PATH] \
  [--screenshots-dir DIR] \
  [--download-timeout SECONDS] \
  [--no-screenshots] [--benchmark] [--profile]
```

Examples:
- Basic: `python main.py 10.1000/j.jmb.2010.01.001`
- Custom downloads dir: `python main.py <DOI> --downloads-path ~/Downloads`
- Longer wait for large files: `python main.py <DOI> --download-timeout 120`
- Skip screenshots if screen capture fails: `python main.py <DOI> --no-screenshots`

Notes:
- If `--downloads-path` is omitted, it defaults to `~/Downloads` or the `DOWNLOADS_DIR` environment variable if set.
- Step screenshots are saved under `pdfcollect/screenshots` by default; override with `--screenshots-dir`.

Troubleshooting screen capture (Linux)
- Ensure you are in a graphical session with `DISPLAY` set. On GNOME/Wayland, prefer an Xorg session for unrestricted capture.
- Allow X access if needed: `xhost +SI:localuser:$USER`.
- If you see GNOME or GdkPixbuf errors, try `--no-screenshots` to proceed and verify the rest of the workflow.
