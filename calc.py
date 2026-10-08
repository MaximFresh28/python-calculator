#!/usr/bin/env python3
"""
Консольный калькулятор с собственным парсером выражений.

Строка не передаётся в eval(): она проходит через лексер, парсер рекурсивного
спуска строит дерево выражения (AST), а вычислитель обходит это дерево.
Поэтому ввод вроде  __import__('os')  не может выполнить произвольный код.

Запуск:
    python calc.py             интерактивный режим
    python calc.py "2+2*2"     вычислить одно выражение
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

Number = Union[int, float]

MAX_INT_DIGITS = 4000            # str(int) в Python 3.11+ ограничен 4300 цифрами
MAX_INT_BITS = int(MAX_INT_DIGITS * 3.3219)
MAX_FACTORIAL = 1000
MAX_CALL_DEPTH = 50
TRIG_EPSILON = 1e-15             # sin(pi) = 1.2e-16 — это шум, а не ответ


class CalcError(Exception):
    """Ошибка ввода или вычисления. pos — позиция символа в исходной строке."""

    def __init__(self, message: str, pos: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.pos = pos


# ═══════════════════════════════════════════════════════════════════════════
#  Лексер: строка → токены
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Token:
    kind: str        # NUM, NAME, OP, LPAREN, RPAREN, COMMA, ASSIGN, END
    value: object
    pos: int


OPERATOR_ALIASES = {'**': '^', '×': '*', '÷': '/', '·': '*'}
TWO_CHAR_OPS = ('**', '//')
ONE_CHAR_OPS = '+-*/%^!×÷·'
PUNCTUATION = {'(': 'LPAREN', ')': 'RPAREN', ',': 'COMMA', '=': 'ASSIGN'}


def _is_digit(ch: str) -> bool:
    return '0' <= ch <= '9'      # str.isdigit() пропустил бы «²» и прочие символы


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
        elif _is_digit(ch) or (ch == '.' and i + 1 < n and _is_digit(text[i + 1])):
            value, end = _read_number(text, i)
            tokens.append(Token('NUM', value, i))
            i = end
        elif ch.isalpha() or ch == '_':
            start = i
            while i < n and (text[i].isalnum() or text[i] == '_'):
                i += 1
            tokens.append(Token('NAME', text[start:i], start))
        elif text[i:i + 2] in TWO_CHAR_OPS:
            op = text[i:i + 2]
            tokens.append(Token('OP', OPERATOR_ALIASES.get(op, op), i))
            i += 2
        elif ch in ONE_CHAR_OPS:
            tokens.append(Token('OP', OPERATOR_ALIASES.get(ch, ch), i))
            i += 1
        elif ch in PUNCTUATION:
            tokens.append(Token(PUNCTUATION[ch], ch, i))
            i += 1
        else:
            raise CalcError(f"Неожиданный символ «{ch}»", i)

    tokens.append(Token('END', None, n))
    return tokens


def _read_number(text: str, start: int) -> Tuple[Number, int]:
    n = len(text)

    if text[start:start + 2].lower() in ('0x', '0b', '0o'):
        i = start + 2
        while i < n and (text[i].isalnum() or text[i] == '_'):
            i += 1
        literal = text[start:i]
        try:
            return int(literal, 0), i
        except ValueError:
            raise CalcError(f"Некорректное число «{literal}»", start) from None

    i = start
    while i < n and (_is_digit(text[i]) or text[i] == '_'):
        i += 1

    is_float = False
    if i < n and text[i] == '.':
        is_float = True
        i += 1
        while i < n and _is_digit(text[i]):
            i += 1

    # Экспонента: 1e3, 2.5E-4. Одиночное «e» после числа — это константа: 2e = 2·e
    if i < n and text[i] in 'eE':
        j = i + 1
        if j < n and text[j] in '+-':
            j += 1
        if j < n and _is_digit(text[j]):
            is_float = True
            i = j
            while i < n and _is_digit(text[i]):
                i += 1

    literal = text[start:i]
    try:
        value: Number = float(literal.replace('_', '')) if is_float else int(literal)
    except ValueError:
        raise CalcError(f"Некорректное число «{literal}»", start) from None

    if isinstance(value, float) and math.isinf(value):
        raise CalcError("Число слишком большое", start)
    return value, i


# ═══════════════════════════════════════════════════════════════════════════
#  AST
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Num:
    value: Number
    pos: int


@dataclass
class Name:
    name: str
    pos: int


@dataclass
class UnaryOp:
    op: str
    operand: 'Node'
    pos: int


@dataclass
class BinOp:
    op: str
    left: 'Node'
    right: 'Node'
    pos: int


@dataclass
class Factorial:
    operand: 'Node'
    pos: int


@dataclass
class Call:
    name: str
    args: List['Node']
    pos: int


Node = Union[Num, Name, UnaryOp, BinOp, Factorial, Call]


@dataclass
class Assign:
    name: str
    expr: Node
    pos: int


@dataclass
class FuncDef:
    name: str
    params: List[str]
    body: Node
    source: str
    pos: int


Statement = Union[Node, Assign, FuncDef]


# ═══════════════════════════════════════════════════════════════════════════
#  Парсер: токены → AST (рекурсивный спуск)
#
#  expr     := term (('+' | '-') term)*
#  term     := unary (('*' | '/' | '//' | '%') unary | <неявное ×> power)*
#  unary    := ('+' | '-') unary | power
#  power    := postfix ('^' unary)?          правая ассоциативность
#  postfix  := primary '!'*
#  primary  := NUM | NAME | NAME '(' args ')' | '(' expr ')'
#
#  Унарный минус слабее степени: -2^2 = -4, как в математике.
# ═══════════════════════════════════════════════════════════════════════════

class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        token = self.tokens[self.i]
        self.i += 1
        return token

    def expect(self, kind: str, message: str) -> Token:
        if self.tok.kind != kind:
            raise CalcError(message, self.tok.pos)
        return self.advance()

    def parse(self) -> Node:
        node = self.expr()
        if self.tok.kind != 'END':
            raise CalcError(self._unexpected(), self.tok.pos)
        return node

    def expr(self) -> Node:
        node = self.term()
        while self.tok.kind == 'OP' and self.tok.value in ('+', '-'):
            op = self.advance()
            node = BinOp(op.value, node, self.term(), op.pos)
        return node

    def term(self) -> Node:
        node = self.unary()
        while True:
            t = self.tok
            if t.kind == 'OP' and t.value in ('*', '/', '//', '%'):
                self.advance()
                node = BinOp(t.value, node, self.unary(), t.pos)
            elif t.kind in ('NAME', 'LPAREN'):
                # Неявное умножение: 2pi, 3(1+2), (1+2)(3+4), 2x
                node = BinOp('*', node, self.power(), t.pos)
            else:
                return node

    def unary(self) -> Node:
        t = self.tok
        if t.kind == 'OP' and t.value in ('+', '-'):
            self.advance()
            return UnaryOp(t.value, self.unary(), t.pos)
        return self.power()

    def power(self) -> Node:
        node = self.postfix()
        t = self.tok
        if t.kind == 'OP' and t.value == '^':
            self.advance()
            return BinOp('^', node, self.unary(), t.pos)
        return node

    def postfix(self) -> Node:
        node = self.primary()
        while self.tok.kind == 'OP' and self.tok.value == '!':
            node = Factorial(node, self.advance().pos)
        return node

    def primary(self) -> Node:
        t = self.tok

        if t.kind == 'NUM':
            self.advance()
            return Num(t.value, t.pos)

        if t.kind == 'NAME':
            self.advance()
            if self.tok.kind != 'LPAREN':
                return Name(t.value, t.pos)
            self.advance()
            args: List[Node] = []
            if self.tok.kind != 'RPAREN':
                args.append(self.expr())
                while self.tok.kind == 'COMMA':
                    self.advance()
                    args.append(self.expr())
            self.expect('RPAREN', "Не хватает закрывающей скобки «)»")
            return Call(t.value, args, t.pos)

        if t.kind == 'LPAREN':
            self.advance()
            node = self.expr()
            self.expect('RPAREN', "Не хватает закрывающей скобки «)»")
            return node

        raise CalcError(self._unexpected(), t.pos)

    def _unexpected(self) -> str:
        t = self.tok
        prev = self.tokens[self.i - 1] if self.i > 0 else None
        if t.kind == 'END':
            return "Выражение оборвалось: ожидалось число, имя или скобка"
        if t.kind == 'RPAREN':
            if prev is not None and prev.kind == 'LPAREN':
                return "Пустые скобки"
            return "Лишняя закрывающая скобка «)»"
        if t.kind == 'NUM':
            return "Пропущен оператор перед числом"
        if t.kind == 'COMMA':
            return "Запятая вне вызова функции"
        if t.kind == 'ASSIGN':
            return "Неожиданный знак «=»"
        return f"Неожиданный символ «{t.value}»"


CONTINUATION_OPS = ('*', '/', '//', '%', '^', '!')


def parse_statement(text: str) -> Statement:
    tokens = tokenize(text)

    # «*2» продолжает предыдущий результат: превращается в «ans*2»
    first = tokens[0]
    if first.kind == 'OP' and first.value in CONTINUATION_OPS:
        tokens.insert(0, Token('NAME', 'ans', first.pos))

    assigns = [k for k, t in enumerate(tokens) if t.kind == 'ASSIGN']
    if not assigns:
        return Parser(tokens).parse()
    if len(assigns) > 1:
        raise CalcError("В строке может быть только один знак «=»", tokens[assigns[1]].pos)

    eq = assigns[0]
    left = tokens[:eq]
    body = Parser(tokens[eq + 1:]).parse()

    if len(left) == 1 and left[0].kind == 'NAME':
        return Assign(left[0].value, body, left[0].pos)

    is_call_shape = (len(left) >= 3 and left[0].kind == 'NAME'
                     and left[1].kind == 'LPAREN' and left[-1].kind == 'RPAREN')
    if is_call_shape:
        inner = left[2:-1]
        for k, t in enumerate(inner):
            if t.kind != ('NAME' if k % 2 == 0 else 'COMMA'):
                raise CalcError("Параметры функции — имена через запятую: f(x, y) = ...", t.pos)
        if inner and inner[-1].kind != 'NAME':
            raise CalcError("После запятой ожидалось имя параметра", left[-1].pos)
        params = [t.value for t in inner if t.kind == 'NAME']
        if len(set(params)) != len(params):
            raise CalcError("Имена параметров повторяются", left[2].pos)
        source = text[left[0].pos:].strip()
        return FuncDef(left[0].value, params, body, source, left[0].pos)

    pos = left[0].pos if left else tokens[eq].pos
    raise CalcError("Слева от «=» должно быть имя: x = 5 или f(x) = x^2", pos)


# ═══════════════════════════════════════════════════════════════════════════
#  Вычислитель
# ═══════════════════════════════════════════════════════════════════════════

CONSTANTS: Dict[str, float] = {
    'pi': math.pi,
    'π': math.pi,
    'e': math.e,
    'tau': math.tau,
    'phi': (1 + math.sqrt(5)) / 2,
    'inf': math.inf,
}

Builtin = Tuple[Callable[..., Number], int, Optional[int]]   # функция, мин. и макс. число аргументов


@dataclass
class Outcome:
    value: Optional[Number]
    message: str


class Calculator:
    def __init__(self) -> None:
        self.variables: Dict[str, Number] = {}
        self.functions: Dict[str, FuncDef] = {}
        self.ans: Optional[Number] = None
        self.degrees = False
        self.history: List[Tuple[str, str]] = []
        self.builtins: Dict[str, Builtin] = self._make_builtins()

    # ── Публичное API ────────────────────────────────────────────────────

    def execute(self, text: str) -> Outcome:
        try:
            stmt = parse_statement(text)
            if isinstance(stmt, FuncDef):
                return self._define_function(stmt)
            if isinstance(stmt, Assign):
                return self._assign(stmt)
            value = self._eval(stmt, {}, 0)
        except RecursionError:
            raise CalcError("Слишком глубокая вложенность выражения") from None

        self.ans = value
        shown = format_number(value)
        self.history.append((text.strip(), shown))
        return Outcome(value, shown)

    def evaluate(self, text: str) -> Number:
        """Удобная обёртка для тестов и использования как библиотеки."""
        value = self.execute(text).value
        if value is None:
            raise CalcError("Строка не является выражением")
        return value

    def reset(self) -> None:
        self.variables.clear()
        self.functions.clear()
        self.history.clear()
        self.ans = None

    # ── Инструкции ───────────────────────────────────────────────────────

    def _assign(self, stmt: Assign) -> Outcome:
        self._check_free_name(stmt.name, stmt.pos, 'variable')
        value = self._eval(stmt.expr, {}, 0)
        self.variables[stmt.name] = value
        self.ans = value
        message = f"{stmt.name} = {format_number(value)}"
        self.history.append((message, format_number(value)))
        return Outcome(value, message)

    def _define_function(self, stmt: FuncDef) -> Outcome:
        self._check_free_name(stmt.name, stmt.pos, 'function')
        for param in stmt.params:
            if param in CONSTANTS or param in self.builtins or param == 'ans':
                raise CalcError(f"«{param}» нельзя использовать как имя параметра", stmt.pos)
        self.functions[stmt.name] = stmt
        self.history.append((stmt.source, "функция"))
        return Outcome(None, f"Функция определена: {stmt.source}")

    def _check_free_name(self, name: str, pos: int, kind: str) -> None:
        if name == 'ans':
            raise CalcError("Имя «ans» зарезервировано за последним результатом", pos)
        if name in CONSTANTS:
            raise CalcError(f"«{name}» — константа, её нельзя переопределить", pos)
        if name in self.builtins:
            raise CalcError(f"«{name}» — встроенная функция, выберите другое имя", pos)
        if kind == 'variable' and name in self.functions:
            raise CalcError(f"Имя «{name}» уже занято функцией", pos)
        if kind == 'function' and name in self.variables:
            raise CalcError(f"Имя «{name}» уже занято переменной", pos)

    # ── Обход дерева ─────────────────────────────────────────────────────

    def _eval(self, node: Node, scope: Dict[str, Number], depth: int) -> Number:
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Name):
            return self._lookup(node.name, node.pos, scope)
        if isinstance(node, UnaryOp):
            value = self._eval(node.operand, scope, depth)
            return -value if node.op == '-' else value
        if isinstance(node, BinOp):
            left = self._eval(node.left, scope, depth)
            right = self._eval(node.right, scope, depth)
            return self._binary(node.op, left, right, node.pos)
        if isinstance(node, Factorial):
            return self._factorial(self._eval(node.operand, scope, depth), node.pos)
        if isinstance(node, Call):
            return self._call(node, scope, depth)
        raise CalcError("Внутренняя ошибка: неизвестный узел выражения")

    def _lookup(self, name: str, pos: int, scope: Dict[str, Number]) -> Number:
        if name in scope:
            return scope[name]
        if name == 'ans':
            if self.ans is None:
                raise CalcError("Предыдущего результата (ans) ещё нет", pos)
            return self.ans
        if name in self.variables:
            return self.variables[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in self.builtins or name in self.functions:
            raise CalcError(f"«{name}» — функция, её вызывают со скобками: {name}(...)", pos)
        raise CalcError(f"Неизвестная переменная «{name}»", pos)

    def _call(self, node: Call, scope: Dict[str, Number], depth: int) -> Number:
        name = node.name

        if name in self.functions and name not in scope:
            fdef = self.functions[name]
            if len(node.args) != len(fdef.params):
                raise CalcError(f"{name}() ожидает аргументов: {len(fdef.params)}, "
                                f"передано: {len(node.args)}", node.pos)
            if depth >= MAX_CALL_DEPTH:
                raise CalcError(f"Слишком глубокая рекурсия в «{name}»", node.pos)
            args = [self._eval(a, scope, depth) for a in node.args]
            return self._eval(fdef.body, dict(zip(fdef.params, args)), depth + 1)

        if name in self.builtins and name not in scope:
            func, lo, hi = self.builtins[name]
            count = len(node.args)
            if count < lo or (hi is not None and count > hi):
                raise CalcError(f"{name}() ожидает {_arity_text(lo, hi)}, передано: {count}", node.pos)
            args = [self._eval(a, scope, depth) for a in node.args]
            try:
                result = func(*args)
            except CalcError as err:
                if err.pos is None:
                    err.pos = node.pos
                raise
            except ZeroDivisionError:
                raise CalcError("Деление на ноль", node.pos) from None
            except OverflowError:
                raise CalcError("Слишком большое число", node.pos) from None
            except (ValueError, TypeError):
                raise CalcError(f"Аргумент вне области определения {name}()", node.pos) from None
            return self._check_result(result, node.pos, *args)

        # x(2+3), когда x — переменная, означает x·(2+3)
        if len(node.args) == 1:
            try:
                value = self._lookup(name, node.pos, scope)
            except CalcError:
                raise CalcError(f"Неизвестная функция «{name}»", node.pos) from None
            return self._binary('*', value, self._eval(node.args[0], scope, depth), node.pos)

        raise CalcError(f"Неизвестная функция «{name}»", node.pos)

    # ── Операции ─────────────────────────────────────────────────────────

    def _binary(self, op: str, a: Number, b: Number, pos: int) -> Number:
        try:
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
            elif op == '/':
                if b == 0:
                    raise CalcError("Деление на ноль", pos)
                # Целые делятся нацело — оставляем результат точным: 6/3 = 2, а не 2.0
                if isinstance(a, int) and isinstance(b, int) and a % b == 0:
                    result = a // b
                else:
                    result = a / b
            elif op == '//':
                if b == 0:
                    raise CalcError("Деление на ноль", pos)
                result = a // b
            elif op == '%':
                if b == 0:
                    raise CalcError("Остаток от деления на ноль", pos)
                result = a % b
            elif op == '^':
                result = self._power(a, b, pos)
            else:
                raise CalcError(f"Неизвестный оператор «{op}»", pos)
        except OverflowError:
            raise CalcError("Слишком большое число", pos) from None
        return self._check_result(result, pos, a, b)

    @staticmethod
    def _power(a: Number, b: Number, pos: int) -> Number:
        if isinstance(a, int) and isinstance(b, int) and b >= 0:
            # Оценка числа цифр заранее: 9^9^9 иначе подвесил бы программу
            if abs(a) > 1 and b * math.log10(abs(a)) > MAX_INT_DIGITS:
                raise CalcError("Слишком большое число", pos)
            return a ** b
        if a == 0 and b < 0:
            raise CalcError("Ноль нельзя возводить в отрицательную степень", pos)
        if a < 0 and isinstance(b, float) and not b.is_integer():
            raise CalcError("Отрицательное число в дробной степени — комплексный результат. "
                            "Для кубического корня есть cbrt()", pos)
        return a ** b

    @staticmethod
    def _factorial(value: Number, pos: Optional[int]) -> int:
        if isinstance(value, float):
            if not value.is_integer():
                raise CalcError("Факториал определён только для целых чисел ≥ 0", pos)
            value = int(value)
        if value < 0:
            raise CalcError("Факториал определён только для целых чисел ≥ 0", pos)
        if value > MAX_FACTORIAL:
            raise CalcError(f"Слишком большой факториал (максимум {MAX_FACTORIAL}!)", pos)
        return math.factorial(value)

    @staticmethod
    def _check_result(result: object, pos: Optional[int], *inputs: Number) -> Number:
        if isinstance(result, complex):
            raise CalcError("Результат — комплексное число", pos)
        if isinstance(result, int) and result.bit_length() > MAX_INT_BITS:
            raise CalcError("Слишком большое число", pos)
        if isinstance(result, float) and math.isinf(result):
            if not any(isinstance(x, float) and math.isinf(x) for x in inputs):
                raise CalcError("Переполнение: результат слишком большой", pos)
        return result  # type: ignore[return-value]

    # ── Встроенные функции ───────────────────────────────────────────────

    def _make_builtins(self) -> Dict[str, Builtin]:
        def to_rad(x: Number) -> float:
            return math.radians(x) if self.degrees else x

        def from_rad(x: float) -> float:
            return math.degrees(x) if self.degrees else x

        def snap(x: float) -> float:
            return 0.0 if abs(x) < TRIG_EPSILON else x

        def sin(x: Number) -> float:
            return snap(math.sin(to_rad(x)))

        def cos(x: Number) -> float:
            return snap(math.cos(to_rad(x)))

        def tan(x: Number) -> float:
            c = cos(x)
            if c == 0:
                raise CalcError("Тангенс в этой точке не определён")
            return snap(math.sin(to_rad(x)) / c)

        def require_int(x: Number, fname: str) -> int:
            if isinstance(x, float):
                if not x.is_integer():
                    raise CalcError(f"{fname}() работает только с целыми числами")
                return int(x)
            return x

        def cbrt(x: Number) -> float:
            return math.copysign(abs(x) ** (1 / 3), x)

        def root(x: Number, n: Number) -> Number:
            if n == 0:
                raise CalcError("Корень нулевой степени не определён")
            if x < 0:
                if float(n).is_integer() and int(n) % 2 == 1:
                    return -((-x) ** (1 / n))
                raise CalcError("Корень чётной степени из отрицательного числа")
            return x ** (1 / n)

        def log(x: Number, base: Number = 10) -> float:
            return math.log(x, base)

        def round_(x: Number, digits: Number = 0) -> Number:
            d = require_int(digits, 'round')
            return round(x, d) if d else round(x)

        def gcd(*args: Number) -> int:
            return math.gcd(*(require_int(a, 'gcd') for a in args))

        def lcm(*args: Number) -> int:
            return math.lcm(*(require_int(a, 'lcm') for a in args))

        def comb(n: Number, k: Number) -> int:
            return math.comb(require_int(n, 'comb'), require_int(k, 'comb'))

        def perm(n: Number, k: Number) -> int:
            return math.perm(require_int(n, 'perm'), require_int(k, 'perm'))

        return {
            'sin': (sin, 1, 1),
            'cos': (cos, 1, 1),
            'tan': (tan, 1, 1),
            'asin': (lambda x: from_rad(math.asin(x)), 1, 1),
            'acos': (lambda x: from_rad(math.acos(x)), 1, 1),
            'atan': (lambda x: from_rad(math.atan(x)), 1, 1),
            'sinh': (math.sinh, 1, 1),
            'cosh': (math.cosh, 1, 1),
            'tanh': (math.tanh, 1, 1),
            'sqrt': (math.sqrt, 1, 1),
            'cbrt': (cbrt, 1, 1),
            'root': (root, 2, 2),
            'exp': (math.exp, 1, 1),
            'ln': (math.log, 1, 1),
            'log': (log, 1, 2),
            'log2': (math.log2, 1, 1),
            'abs': (abs, 1, 1),
            'floor': (math.floor, 1, 1),
            'ceil': (math.ceil, 1, 1),
            'round': (round_, 1, 2),
            'min': (min, 1, None),
            'max': (max, 1, None),
            'sum': (lambda *a: sum(a), 1, None),
            'avg': (lambda *a: sum(a) / len(a), 1, None),
            'gcd': (gcd, 1, None),
            'lcm': (lcm, 1, None),
            'comb': (comb, 2, 2),
            'perm': (perm, 2, 2),
            'hypot': (math.hypot, 1, None),
            'fact': (lambda x: self._factorial(x, None), 1, 1),
            'deg': (math.degrees, 1, 1),
            'rad': (math.radians, 1, 1),
        }


def _arity_text(lo: int, hi: Optional[int]) -> str:
    if hi is None:
        return f"не меньше {lo} аргумента" if lo == 1 else f"не меньше {lo} аргументов"
    if lo == hi:
        return f"аргументов: {lo}"
    return f"от {lo} до {hi} аргументов"


def format_number(x: Number) -> str:
    if isinstance(x, int):
        return str(x)
    if math.isnan(x):
        return "не число"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    if x == 0:
        return "0"                       # заодно убирает «-0»
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.12g}"                   # 0.1 + 0.2 → 0.3, а не 0.30000000000000004


# ═══════════════════════════════════════════════════════════════════════════
#  Интерфейс командной строки
# ═══════════════════════════════════════════════════════════════════════════

try:
    import readline  # noqa: F401  — стрелки и история ввода в Linux/macOS
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


def _enable_ansi() -> bool:
    if not sys.stdout.isatty() or 'NO_COLOR' in os.environ:
        return False
    if os.name == 'nt':
        os.system('')                    # включает обработку ANSI в консоли Windows 10+
    return True


COLOR = False


def paint(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if COLOR else text


def prompt_text(calc: Calculator) -> Tuple[str, int]:
    """Возвращает строку приглашения и её видимую длину (для стрелки под ошибкой)."""
    label = "deg › " if calc.degrees else "› "
    if not COLOR:
        return label, len(label)
    start, end = ("\001", "\002") if HAS_READLINE else ("", "")
    colored = f"{start}\x1b[1;36m{end}{label}{start}\x1b[0m{end}"
    return colored, len(label)


HELP = """
Операторы    + - * /    // целочисленное    % остаток    ^ или ** степень    ! факториал
Приоритеты   2+3*4 = 14    2^3^2 = 512    -2^2 = -4    2^-1 = 0.5
Неявное ×    2pi    3(1+2)    (1+2)(3+4)    2x
Числа        1.5   .5   1e-3   1_000_000   0xFF   0b1010   0o17
Константы    pi (π)   e   tau   phi   inf

Переменные   x = 5          ans — последний результат
Продолжение  *2   /3   ^2   — строка с оператора продолжает от ans
Функции      f(x) = x^2 + 1          dist(a, b) = sqrt(a^2 + b^2)

Встроенные   sin cos tan asin acos atan sinh cosh tanh
             sqrt cbrt root(x, n) exp ln log(x[, основание]) log2
             abs floor ceil round(x[, знаков]) min max sum avg
             gcd lcm comb perm hypot fact deg rad

Команды      :deg / :rad  режим углов     :vars  переменные     :funcs  функции
             :history     история         :clear сброс          :q      выход
"""

EXIT_COMMANDS = {'q', 'quit', 'exit', 'выход'}


def run_command(calc: Calculator, line: str) -> bool:
    """Выполняет служебную команду. Возвращает False, если пора выходить."""
    cmd = line.strip().lstrip(':').lower()

    if cmd in EXIT_COMMANDS:
        return False
    if cmd in ('help', 'h', '?', 'помощь'):
        print(HELP)
    elif cmd == 'deg':
        calc.degrees = True
        print(paint("Углы в градусах", '33'))
    elif cmd == 'rad':
        calc.degrees = False
        print(paint("Углы в радианах", '33'))
    elif cmd == 'vars':
        if calc.ans is not None:
            print(f"  ans = {format_number(calc.ans)}")
        for name, value in calc.variables.items():
            print(f"  {name} = {format_number(value)}")
        if calc.ans is None and not calc.variables:
            print(paint("  Переменных пока нет", '2'))
    elif cmd == 'funcs':
        for fdef in calc.functions.values():
            print(f"  {fdef.source}")
        if not calc.functions:
            print(paint("  Функций пока нет. Пример: f(x) = x^2 + 1", '2'))
    elif cmd == 'history':
        for k, (expr, result) in enumerate(calc.history[-20:], 1):
            print(f"  {k:>2}. {expr}  {paint('→ ' + result, '2')}")
        if not calc.history:
            print(paint("  История пуста", '2'))
    elif cmd == 'clear':
        calc.reset()
        print(paint("Переменные, функции и история очищены", '33'))
    else:
        print(paint(f"Неизвестная команда «{line.strip()}». Список команд — :help", '31'))
    return True


def is_command(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(':') or stripped.lower() in EXIT_COMMANDS | {'help', '?', 'помощь'}


def print_error(err: CalcError, caret_offset: Optional[int]) -> None:
    if err.pos is not None and caret_offset is not None:
        print(" " * (caret_offset + err.pos) + paint("^", '1;31'))
    print(paint("Ошибка: ", '1;31') + err.message)


def print_result(outcome: Outcome) -> None:
    if outcome.value is None or outcome.message != format_number(outcome.value):
        print(paint(outcome.message, '32'))          # определение функции или присваивание
        return
    print(paint("= ", '2') + paint(outcome.message, '1;32'))
    digits = outcome.message.lstrip('-')
    if digits.isdigit() and len(digits) > 60:
        print(paint(f"  ({len(digits)} цифр)", '2'))


def repl(calc: Calculator) -> None:
    print(paint("Калькулятор", '1;36') + paint("  ·  :help — справка, :q — выход", '2'))
    while True:
        prompt, prompt_len = prompt_text(calc)
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line.strip():
            continue
        if is_command(line):
            if not run_command(calc, line):
                break
            continue

        try:
            print_result(calc.execute(line))
        except CalcError as err:
            print_error(err, prompt_len)


def main(argv: Optional[List[str]] = None) -> int:
    global COLOR
    args = sys.argv[1:] if argv is None else argv
    COLOR = _enable_ansi()
    calc = Calculator()

    if args:
        expression = " ".join(args)
        try:
            print(calc.execute(expression).message)
            return 0
        except CalcError as err:
            print("  " + expression)
            print_error(err, 2)
            return 1

    repl(calc)
    return 0


if __name__ == '__main__':
    sys.exit(main())
