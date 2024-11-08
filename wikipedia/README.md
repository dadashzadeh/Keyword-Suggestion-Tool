# Wikipedia Interactive Graph Generator

This project creates an interactive graph of related Wikipedia articles for a given query using the Wikipedia API. It searches for the main topics and their connections up to a specified depth, and saves the data as both an interactive HTML graph and a detailed Excel file.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
---

## Overview

This Python project fetches and visualizes Wikipedia search results and their related pages using an interactive graph. The tool is built with asynchronous API requests for efficient data fetching and stores the results in two main formats:
- An interactive HTML graph (using `pyvis`)
- A data file with edges and nodes stored in an Excel spreadsheet

## Features

- **Asynchronous Wikipedia API fetching**: Quickly retrieve search results and related pages.
- **Graph Depth**: Customize the depth of related page exploration.
- **Interactive Graph Generation**: View relationships between Wikipedia pages in a browser with `pyvis`.
- **Excel Export**: Save nodes and edges to Excel for easy data analysis.
