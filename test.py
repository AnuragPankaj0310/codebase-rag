from tree_sitter import Language
from tree_sitter import Parser
import tree_sitter_python

PY_LANGUAGE = Language(
    tree_sitter_python.language()
)

parser = Parser(
    PY_LANGUAGE
)

print("Parser initialized!")