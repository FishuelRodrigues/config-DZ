#!/usr/bin/env python3
"""
Конфигурационный преобразователь (вариант 16)
Преобразует конфигурационный язык в XML
"""

import sys
import re
import json
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Union, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# Классы для представления AST (Abstract Syntax Tree)
# ============================================================================

class ValueType(Enum):
    """Типы значений в конфигурационном языке"""
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    DICTIONARY = "dict"
    CONSTANT = "constant"

@dataclass
class ASTNode:
    """Базовый класс для узлов AST"""
    pass

@dataclass
class Number(ASTNode):
    value: Union[int, float]
    
    def __init__(self, val: str):
        # Поддержка разных форматов чисел: целые, вещественные, научная запись
        try:
            self.value = int(val)
        except ValueError:
            self.value = float(val)

@dataclass
class String(ASTNode):
    value: str
    
    def __init__(self, val: str):
        # Убираем внешние кавычки из строки
        self.value = val[1:-1] if val.startswith('"') and val.endswith('"') else val

@dataclass
class Array(ASTNode):
    elements: List[ASTNode]

@dataclass
class Dictionary(ASTNode):
    pairs: Dict[str, ASTNode]

@dataclass
class Constant(ASTNode):
    name: str

# ============================================================================
# Лексический анализатор (Tokenizer)
# ============================================================================

class TokenType(Enum):
    """Типы токенов"""
    NUMBER = "NUMBER"
    STRING = "STRING"
    NAME = "NAME"
    KEYWORD_SET = "SET"
    KEYWORD_TABLE = "TABLE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    EQUAL = "EQUAL"
    COMMA = "COMMA"
    NEWLINE = "NEWLINE"
    COMMENT_START = "COMMENT_START"
    COMMENT_END = "COMMENT_END"
    AT = "AT"
    PLUS = "PLUS"
    PERCENT = "PERCENT"
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    """Лексический анализатор"""
    
    # Регулярные выражения для токенов
    TOKEN_SPECS = [
        (TokenType.COMMENT_START, r'{\[\s*-\s*'),
        (TokenType.COMMENT_END, r'--}}'),
        (TokenType.NUMBER, r'-?(\d+|\d+\.\d*|\.\d+)([eE][-+]?\d+)?'),
        (TokenType.STRING, r'"([^"\\]|\\.)*"'),
        (TokenType.KEYWORD_SET, r'set\b'),
        (TokenType.KEYWORD_TABLE, r'table\b'),
        (TokenType.NAME, r'[_a-zA-Z]+'),
        (TokenType.LBRACKET, r'\['),
        (TokenType.RBRACKET, r'\]'),
        (TokenType.LPAREN, r'\('),
        (TokenType.RPAREN, r'\)'),
        (TokenType.LBRACE, r'\{'),
        (TokenType.RBRACE, r'\}'),
        (TokenType.EQUAL, r'='),
        (TokenType.COMMA, r','),
        (TokenType.AT, r'@'),
        (TokenType.PLUS, r'\+'),
        (TokenType.PERCENT, r'%'),
        (TokenType.NEWLINE, r'\n'),
        ('SKIP', r'[ \t\r]+'),
        ('MISMATCH', r'.'),
    ]
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Компилирует регулярные выражения"""
        patterns = []
        for tok_type, pattern in self.TOKEN_SPECS:
            if tok_type != 'SKIP' and tok_type != 'MISMATCH':
                patterns.append(f'(?P<{tok_type.value}>{pattern})')
            elif tok_type == 'SKIP':
                patterns.append(f'(?:{pattern})')
            else:
                patterns.append(f'(?P<MISMATCH>{pattern})')
        
        self.pattern = re.compile('|'.join(patterns), re.MULTILINE)
    
    def tokenize(self) -> List[Token]:
        """Разбивает текст на токены"""
        for match in self.pattern.finditer(self.text):
            kind = match.lastgroup
            value = match.group()
            
            # Пропускаем пробельные символы
            if kind == 'SKIP':
                self.column += len(value)
                continue
            
            # Обработка несоответствия
            if kind == 'MISMATCH':
                raise SyntaxError(f"Неожиданный символ '{value}' на строке {self.line}, колонка {self.column}")
            
            # Определяем тип токена
            token_type = TokenType(kind)
            
            # Создаем токен
            token = Token(
                type=token_type,
                value=value,
                line=self.line,
                column=self.column
            )
            
            self.tokens.append(token)
            
            # Обновляем позицию
            if token_type == TokenType.NEWLINE:
                self.line += 1
                self.column = 1
            else:
                self.column += len(value)
            
            # Пропускаем комментарии
            if token_type == TokenType.COMMENT_START:
                self._skip_comment()
        
        # Добавляем маркер конца файла
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
    
    def _skip_comment(self):
        """Пропускает многострочные комментарии"""
        # Ищем конец комментария
        comment_end = re.search(r'--}}', self.text[self.pos:])
        if not comment_end:
            raise SyntaxError("Незавершенный комментарий")
        
        # Пропускаем текст комментария
        end_pos = self.pos + comment_end.end()
        skipped = self.text[self.pos:end_pos]
        
        # Считаем новые строки в комментарии
        newlines = skipped.count('\n')
        self.line += newlines
        
        if newlines > 0:
            # Если были новые строки, колонка начинается с 1
            last_newline = skipped.rfind('\n')
            self.column = len(skipped) - last_newline
        else:
            self.column += len(skipped)
        
        self.pos = end_pos

# ============================================================================
# Синтаксический анализатор (Parser)
# ============================================================================

class Parser:
    """Синтаксический анализатор (рекурсивный спуск)"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None
        self.constants: Dict[str, ASTNode] = {}
    
    def error(self, message: str):
        """Генерирует ошибку синтаксиса"""
        raise SyntaxError(f"{message} на строке {self.current_token.line}")
    
    def eat(self, token_type: TokenType):
        """Потребляет токен ожидаемого типа"""
        if self.current_token.type == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = Token(TokenType.EOF, '', 0, 0)
        else:
            self.error(f"Ожидался {token_type}, получен {self.current_token.type}")
    
    def parse(self) -> List[ASTNode]:
        """Основной метод разбора"""
        results = []
        
        while self.current_token.type != TokenType.EOF:
            # Пропускаем пустые строки
            if self.current_token.type == TokenType.NEWLINE:
                self.eat(TokenType.NEWLINE)
                continue
            
            # Определяем начало конструкции
            if self.current_token.type == TokenType.KEYWORD_SET:
                results.append(self.parse_constant_declaration())
            elif self.current_token.type == TokenType.KEYWORD_TABLE:
                results.append(self.parse_dictionary())
            elif self.current_token.type == TokenType.LBRACE:
                results.append(self.parse_array())
            else:
                self.error("Ожидалось объявление константы, словаря или массива")
            
            # Пропускаем завершающие переводы строк
            while self.current_token.type == TokenType.NEWLINE:
                self.eat(TokenType.NEWLINE)
        
        return results
    
    def parse_constant_declaration(self) -> ASTNode:
        """Разбор объявления константы: set имя = значение"""
        self.eat(TokenType.KEYWORD_SET)  # set
        name = self.current_token.value
        self.eat(TokenType.NAME)  # имя
        self.eat(TokenType.EQUAL)  # =
        
        # Парсим значение
        value = self.parse_value()
        
        # Сохраняем константу
        self.constants[name] = value
        
        # Создаем узел константы
        return Constant(name)
    
    def parse_dictionary(self) -> Dictionary:
        """Разбор словаря: table([ имя = значение, ... ])"""
        self.eat(TokenType.KEYWORD_TABLE)  # table
        self.eat(TokenType.LPAREN)  # (
        self.eat(TokenType.LBRACKET)  # [
        
        pairs = {}
        first = True
        
        while self.current_token.type != TokenType.RBRACKET:
            if not first:
                self.eat(TokenType.COMMA)
            first = False
            
            # Парсим пару ключ-значение
            key = self.current_token.value
            self.eat(TokenType.NAME)
            self.eat(TokenType.EQUAL)
            value = self.parse_value()
            pairs[key] = value
        
        self.eat(TokenType.RBRACKET)  # ]
        self.eat(TokenType.RPAREN)  # )
        
        return Dictionary(pairs)
    
    def parse_array(self) -> Array:
        """Разбор массива: #( значение, значение, ... )"""
        self.eat(TokenType.LBRACE)  # {
        
        elements = []
        first = True
        
        while self.current_token.type != TokenType.RBRACE:
            if not first:
                # В массивах значения разделены точкой с запятой
                if self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)
            first = False
            
            # Парсим значение
            value = self.parse_value()
            elements.append(value)
        
        self.eat(TokenType.RBRACE)  # }
        
        return Array(elements)
    
    def parse_value(self) -> ASTNode:
        """Парсинг значения"""
        token = self.current_token
        
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Number(token.value)
        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return String(token.value)
        elif token.type == TokenType.NAME:
            # Это может быть имя константы
            name = token.value
            self.eat(TokenType.NAME)
            
            # Проверяем, объявлена ли константа
            if name in self.constants:
                return Constant(name)
            else:
                # Если это не константа, возможно это строка без кавычек
                return String(name)
        elif token.type == TokenType.KEYWORD_TABLE:
            return self.parse_dictionary()
        elif token.type == TokenType.LBRACE:
            return self.parse_array()
        elif token.type == TokenType.AT:
            return self.parse_constant_expression()
        else:
            self.error(f"Неожиданный токен в значении: {token.type}")
    
    def parse_constant_expression(self) -> ASTNode:
        """Разбор константного выражения: @{выражение}"""
        self.eat(TokenType.AT)  # @
        self.eat(TokenType.LBRACE)  # {
        
        # Парсим инфиксное выражение
        result = self.parse_infix_expression()
        
        self.eat(TokenType.RBRACE)  # }
        
        return result
    
    def parse_infix_expression(self) -> ASTNode:
        """Парсинг инфиксного выражения"""
        # Парсим первый терм
        left = self.parse_term()
        
        # Обрабатываем операции
        while self.current_token.type in (TokenType.PLUS, TokenType.PERCENT):
            op = self.current_token.type
            self.eat(op)
            right = self.parse_term()
            
            # Создаем узел операции
            if op == TokenType.PLUS:
                # Для сложения вычисляем значения
                left_val = self._evaluate_node(left)
                right_val = self._evaluate_node(right)
                left = Number(str(left_val + right_val))
            elif op == TokenType.PERCENT:
                # Для остатка от деления
                left_val = self._evaluate_node(left)
                right_val = self._evaluate_node(right)
                left = Number(str(left_val % right_val))
        
        return left
    
    def parse_term(self) -> ASTNode:
        """Парсинг терма"""
        token = self.current_token
        
        if token.type == TokenType.NAME:
            # Проверяем, является ли это вызовом функции
            if token.value == 'print':
                return self.parse_function_call()
            elif token.value == 'mod':
                return self.parse_function_call()
            else:
                # Это имя константы или переменной
                name = token.value
                self.eat(TokenType.NAME)
                
                if name in self.constants:
                    return Constant(name)
                else:
                    self.error(f"Неизвестная константа: {name}")
        elif token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Number(token.value)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            expr = self.parse_infix_expression()
            self.eat(TokenType.RPAREN)
            return expr
        else:
            self.error(f"Неожиданный токен в терме: {token.type}")
    
    def parse_function_call(self) -> ASTNode:
        """Парсинг вызова функции"""
        func_name = self.current_token.value
        self.eat(TokenType.NAME)
        self.eat(TokenType.LPAREN)
        
        if func_name == 'print':
            # Функция print с одним аргументом
            arg = self.parse_infix_expression()
            self.eat(TokenType.RPAREN)
            
            # Вычисляем аргумент и создаем строку
            arg_val = self._evaluate_node(arg)
            return String(str(arg_val))
        
        elif func_name == 'mod':
            # Функция mod с двумя аргументами
            arg1 = self.parse_infix_expression()
            self.eat(TokenType.COMMA)
            arg2 = self.parse_infix_expression()
            self.eat(TokenType.RPAREN)
            
            # Вычисляем аргументы
            arg1_val = self._evaluate_node(arg1)
            arg2_val = self._evaluate_node(arg2)
            
            # Вычисляем остаток от деления
            if arg2_val == 0:
                self.error("Деление на ноль в функции mod()")
            
            return Number(str(arg1_val % arg2_val))
    
    def _evaluate_node(self, node: ASTNode) -> Any:
        """Вычисляет значение узла"""
        if isinstance(node, Number):
            return node.value
        elif isinstance(node, String):
            return node.value
        elif isinstance(node, Constant):
            const_node = self.constants[node.name]
            return self._evaluate_node(const_node)
        else:
            self.error(f"Невозможно вычислить узел типа {type(node).__name__}")

# ============================================================================
# Вычислитель константных выражений
# ============================================================================

class ConstantEvaluator:
    """Вычисляет константные выражения и заменяет константы их значениями"""
    
    def __init__(self, constants: Dict[str, ASTNode]):
        self.constants = constants
    
    def evaluate(self, node: ASTNode) -> ASTNode:
        """Рекурсивно вычисляет значения констант в AST"""
        if isinstance(node, Constant):
            # Заменяем константу ее значением
            if node.name in self.constants:
                return self.evaluate(self.constants[node.name])
            else:
                raise ValueError(f"Неопределенная константа: {node.name}")
        
        elif isinstance(node, Dictionary):
            # Рекурсивно обрабатываем все значения в словаре
            new_pairs = {}
            for key, value in node.pairs.items():
                new_pairs[key] = self.evaluate(value)
            return Dictionary(new_pairs)
        
        elif isinstance(node, Array):
            # Рекурсивно обрабатываем все элементы массива
            new_elements = [self.evaluate(elem) for elem in node.elements]
            return Array(new_elements)
        
        elif isinstance(node, (Number, String)):
            # Числа и строки не требуют вычисления
            return node
        
        else:
            raise TypeError(f"Неизвестный тип узла: {type(node)}")

# ============================================================================
# Преобразователь в XML
# ============================================================================

class XMLConverter:
    """Преобразует AST в XML"""
    
    @staticmethod
    def to_xml(node: ASTNode) -> ET.Element:
        """Конвертирует узел AST в XML элемент"""
        if isinstance(node, Dictionary):
            return XMLConverter._dict_to_xml(node)
        elif isinstance(node, Array):
            return XMLConverter._array_to_xml(node)
        elif isinstance(node, Number):
            return XMLConverter._number_to_xml(node)
        elif isinstance(node, String):
            return XMLConverter._string_to_xml(node)
        else:
            raise TypeError(f"Неподдерживаемый тип узла для XML: {type(node)}")
    
    @staticmethod
    def _dict_to_xml(dictionary: Dictionary) -> ET.Element:
        """Конвертирует словарь в XML"""
        dict_elem = ET.Element("dict")
        
        for key, value in dictionary.pairs.items():
            pair_elem = ET.SubElement(dict_elem, "pair")
            
            # Ключ
            key_elem = ET.SubElement(pair_elem, "key")
            key_elem.text = key
            
            # Значение
            value_elem = ET.SubElement(pair_elem, "value")
            value_elem.append(XMLConverter.to_xml(value))
        
        return dict_elem
    
    @staticmethod
    def _array_to_xml(array: Array) -> ET.Element:
        """Конвертирует массив в XML"""
        array_elem = ET.Element("array")
        
        for element in array.elements:
            value_elem = ET.SubElement(array_elem, "value")
            value_elem.append(XMLConverter.to_xml(element))
        
        return array_elem
    
    @staticmethod
    def _number_to_xml(number: Number) -> ET.Element:
        """Конвертирует число в XML"""
        num_elem = ET.Element("number")
        num_elem.text = str(number.value)
        return num_elem
    
    @staticmethod
    def _string_to_xml(string: String) -> ET.Element:
        """Конвертирует строку в XML"""
        str_elem = ET.Element("string")
        str_elem.text = string.value
        return str_elem
    
    @staticmethod
    def prettify(xml_string: str) -> str:
        """Форматирует XML для красивого вывода"""
        parsed = minidom.parseString(xml_string)
        return parsed.toprettyxml(indent="  ")

# ============================================================================
# Основной класс конвертера
# ============================================================================

class ConfigConverter:
    """Основной класс конвертера конфигураций"""
    
    def __init__(self):
        self.constants = {}
    
    def convert(self, input_text: str) -> str:
        """Основной метод конвертации"""
        # Лексический анализ
        lexer = Lexer(input_text)
        tokens = lexer.tokenize()
        
        # Синтаксический анализ
        parser = Parser(tokens)
        ast_nodes = parser.parse()
        
        # Получаем константы из парсера
        self.constants = parser.constants
        
        # Вычисляем константные выражения
        evaluator = ConstantEvaluator(self.constants)
        evaluated_nodes = [evaluator.evaluate(node) for node in ast_nodes 
                          if not isinstance(node, Constant)]
        
        # Создаем корневой XML элемент
        root_elem = ET.Element("config")
        
        # Конвертируем каждый узел в XML
        for node in evaluated_nodes:
            root_elem.append(XMLConverter.to_xml(node))
        
        # Преобразуем в строку
        xml_str = ET.tostring(root_elem, encoding='unicode')
        
        # Форматируем вывод
        return XMLConverter.prettify(xml_str)

# ============================================================================
# Примеры конфигураций (для тестирования)
# ============================================================================

EXAMPLE_CONFIGS = [
    # Пример 1: Конфигурация сервера
    """
set port = 8080
set host = "localhost"
set timeout = 300

table([
    server_name = "api_server",
    max_connections = 1000,
    enabled_features = #{ "ssl", "compression", "caching" },
    connection_settings = table([
        keep_alive = true,
        buffer_size = 8192,
        timeout = @{timeout + 100}
    ])
])
""",

    # Пример 2: Конфигурация игры
    """
set player_count = 4
set board_size = 10
set start_health = 100

table([
    game_name = "Space Adventure",
    difficulty = "medium",
    players = #{ "player1", "player2", "player3", "player4" },
    game_settings = table([
        grid_size = board_size,
        max_turns = 500,
        player_health = @{start_health * player_count % 150},
        allowed_actions = #{ "move", "attack", "defend", "use_item" }
    ]),
    victory_condition = "destroy_all_opponents"
])
""",

    # Пример 3: Конфигурация UI
    """
set default_font_size = 14
set primary_color = "#3498db"
set max_width = 1200

table([
    theme_name = "Modern Light",
    colors = table([
        primary = primary_color,
        secondary = "#2ecc71",
        background = "#ffffff",
        text = "#2c3e50"
    ]),
    typography = table([
        font_family = "Arial, sans-serif",
        base_size = default_font_size,
        heading_sizes = #{ 24, 20, 16, 14, 12, 10 }
    ]),
    layout = table([
        container_width = max_width,
        gutter = 20,
        columns = 12,
        breakpoints = #{ 768, 992, 1200 }
    ])
])
"""
]

# ============================================================================
# Тесты
# ============================================================================

import unittest

class TestConfigConverter(unittest.TestCase):
    """Тесты для конвертера конфигураций"""
    
    def test_lexer_basic(self):
        """Тест лексического анализатора"""
        text = 'set x = 42'
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        token_types = [t.type for t in tokens[:-1]]  # Исключаем EOF
        
        expected = [
            TokenType.KEYWORD_SET,
            TokenType.NAME,
            TokenType.EQUAL,
            TokenType.NUMBER
        ]
        
        self.assertEqual(token_types, expected)
    
    def test_parser_constant(self):
        """Тест парсера для констант"""
        text = 'set pi = 3.14159'
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        nodes = parser.parse()
        
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Constant)
        self.assertEqual(nodes[0].name, 'pi')
        self.assertIn('pi', parser.constants)
    
    def test_parser_dictionary(self):
        """Тест парсера для словарей"""
        text = 'table([name = "test", value = 42])'
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        nodes = parser.parse()
        
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Dictionary)
        self.assertEqual(len(nodes[0].pairs), 2)
    
    def test_constant_evaluation(self):
        """Тест вычисления константных выражений"""
        text = """
set a = 10
set b = 20
table([result = @{a + b}])
"""
        
        converter = ConfigConverter()
        result = converter.convert(text)
        
        # Проверяем, что результат содержит вычисленное значение
        self.assertIn("<number>30</number>", result)
    
    def test_function_print(self):
        """Тест функции print()"""
        text = """
set value = 42
table([message = @{print(value)}])
"""
        
        converter = ConfigConverter()
        result = converter.convert(text)
        
        self.assertIn("<string>42</string>", result)
    
    def test_function_mod(self):
        """Тест функции mod()"""
        text = """
set a = 17
set b = 5
table([remainder = @{mod(a, b)}])
"""
        
        converter = ConfigConverter()
        result = converter.convert(text)
        
        self.assertIn("<number>2</number>", result)
    
    def test_array_parsing(self):
        """Тест парсинга массивов"""
        text = '#{1, 2, 3, "four", "five"}'
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        nodes = parser.parse()
        
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], Array)
        self.assertEqual(len(nodes[0].elements), 5)
    
    def test_comment_skipping(self):
        """Тест пропуска комментариев"""
        text = """{[] -
Это многострочный
комментарий
--}}
set x = 42
"""
        
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        token_values = [t.value for t in tokens if t.type != TokenType.EOF]
        
        # Проверяем, что комментарий был пропущен
        self.assertNotIn("{[] -", token_values)
        self.assertNotIn("--}}", token_values)
        self.assertIn("set", token_values)
        self.assertIn("x", token_values)
        self.assertIn("42", token_values)
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Неправильный синтаксис
        text = 'set x = '
        
        with self.assertRaises(SyntaxError):
            lexer = Lexer(text)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            parser.parse()
    
    def test_complex_config(self):
        """Тест сложной конфигурации"""
        converter = ConfigConverter()
        
        for i, config in enumerate(EXAMPLE_CONFIGS, 1):
            try:
                result = converter.convert(config)
                # Проверяем, что XML валиден
                ET.fromstring(result)
                print(f"✓ Пример {i} успешно сконвертирован")
            except Exception as e:
                self.fail(f"Ошибка при конвертации примера {i}: {str(e)}")

# ============================================================================
# Командный интерфейс
# ============================================================================

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Конвертер конфигурационного языка в XML (вариант 16)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s input.conf output.xml
  %(prog)s --test
  %(prog)s --example 1
        
Примеры конфигураций:
  1. Конфигурация сервера
  2. Конфигурация игры
  3. Конфигурация UI
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Входной файл с конфигурацией'
    )
    
    parser.add_argument(
        'output',
        nargs='?',
        help='Выходной XML файл'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Запустить тесты'
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3],
        help='Показать пример конфигурации'
    )
    
    args = parser.parse_args()
    
    # Запуск тестов
    if args.test:
        print("Запуск тестов...")
        unittest.main(argv=[''], exit=False)
        return
    
    # Показ примера
    if args.example:
        if 1 <= args.example <= len(EXAMPLE_CONFIGS):
            print(f"\nПример {args.example}:\n")
            print(EXAMPLE_CONFIGS[args.example - 1])
            
            # Показываем результат конвертации
            try:
                converter = ConfigConverter()
                result = converter.convert(EXAMPLE_CONFIGS[args.example - 1])
                print("\nРезультат конвертации в XML:\n")
                print(result[:500] + "..." if len(result) > 500 else result)
            except Exception as e:
                print(f"\nОшибка при конвертации: {str(e)}")
        else:
            print(f"Пример {args.example} не найден")
        return
    
    # Проверка аргументов
    if not args.input or not args.output:
        parser.print_help()
        print("\nОшибка: необходимо указать входной и выходной файлы")
        sys.exit(1)
    
    # Чтение входного файла
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл '{args.input}' не найден")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        sys.exit(1)
    
    # Конвертация
    try:
        converter = ConfigConverter()
        xml_output = converter.convert(input_text)
        
        # Запись в файл
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(xml_output)
        
        print(f"Успешно сконвертировано в '{args.output}'")
        
    except SyntaxError as e:
        print(f"Ошибка синтаксиса: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при конвертации: {str(e)}")
        sys.exit(1)

# ============================================================================
# Точка входа
# ============================================================================

if __name__ == "__main__":
    main()