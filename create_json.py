import json

data = [
    {
        "FolderPath": "Category1",
        "FileName": "sample1.pdf",
        "PdfTextContent": "This is a sample text for Category1. It contains information related to the first category."
    },
    {
        "FolderPath": "Category1",
        "FileName": "sample2.pdf",
        "PdfTextContent": "Another example document for Category1. More information about this category."
    },
    {
        "FolderPath": "Category2",
        "FileName": "sample3.pdf",
        "PdfTextContent": "This is a document from Category2. It contains different information."
    },
    {
        "FolderPath": "Category2",
        "FileName": "sample4.pdf",
        "PdfTextContent": "Another Category2 document with specific information about this category."
    },
    {
        "FolderPath": "Category3",
        "FileName": "sample5.pdf",
        "PdfTextContent": "Document for Category3 with completely different information."
    },
    {
        "FolderPath": "Category3",
        "FileName": "sample6.pdf",
        "PdfTextContent": "Another example from Category3 explaining the specifics of this category."
    }
]

with open('L1Files.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("L1Files.json has been created successfully.")
