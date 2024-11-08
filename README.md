# Autocomplete Suggestions Graph Generator

This Python script fetches autocomplete suggestions from Google's public autocomplete API based on a user-defined query and constructs a graph representing these suggestions. It visualizes the relationships between the original query and its autocomplete suggestions using the `pyvis` library and exports the results to an Excel file.

## Features

- Fetches autocomplete suggestions for a given query.
- Constructs a directed graph of suggestions using NetworkX.
- Visualizes the graph using Pyvis and saves it as an interactive HTML file.
- Exports the nodes and edges of the graph to an Excel file with separate sheets.

## Requirements

To run this script, you need to have Python installed along with the following libraries:

- `httpx` for making HTTP requests.
- `pyvis` for visualizing the graph.
- `networkx` for creating and managing the graph structure.
- `pandas` for handling data and exporting to Excel.
- `datetime` for timestamping the output files.

You can install the required libraries using pip:

```bash
pip install httpx pyvis networkx pandas
