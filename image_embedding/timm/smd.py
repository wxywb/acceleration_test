import snakemd
doc = snakemd.new_doc("README")
from snakemd import Document,Table

doc = Document("README")

doc.output_page()

doc.add_header("Why Use SnakeMD?")

p = doc.add_paragraph(
  """
  SnakeMD is a library for generating markdown, and here's
  why you might choose to use it:
  """
)
doc.add_unordered_list([
    "SnakeMD makes it easy to create markdown files.",
    "SnakeMD has been used to back of The Renegade Coder projects."
])

doc.add_table(
        ["Height (cm)", "Weight (kg)", "Age (y)"],
        [
            ['150', '70', '21'],
            ['164', '75', '19'],
            ['181', '87', '40']
        ],
        [Table.Align.LEFT, Table.Align.CENTER, Table.Align.RIGHT],
        0
    )
p.insert_link("SnakeMD", "https://snakemd.therenegadecoder.com")

print(type(doc))
doc.output_page('./')
