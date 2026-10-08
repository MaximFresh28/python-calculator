"""Тесты калькулятора.  Запуск:  python -m unittest -v"""

import math
import unittest

from calc import CalcError, Calculator, format_number


class CalcTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.calc = Calculator()

    def ev(self, text: str):
        return self.calc.evaluate(text)

    def shown(self, text: str) -> str:
        return self.calc.execute(text).message

    def assertCalcError(self, text: str, fragment: str = "") -> CalcError:
        with self.assertRaises(CalcError) as ctx:
            self.calc.execute(text)
        if fragment:
            self.assertIn(fragment, ctx.exception.message)
        return ctx.exception


class TestArithmetic(CalcTestCase):
    def test_precedence(self):
        self.assertEqual(self.ev("2+2*2"), 6)
        self.assertEqual(self.ev("(2+2)*2"), 8)
        self.assertEqual(self.ev("10-4-3"), 3)

    def test_power_is_right_associative(self):
        self.assertEqual(self.ev("2^3^2"), 512)
        self.assertEqual(self.ev("2**3"), 8)

    def test_unary_minus_binds_weaker_than_power(self):
        self.assertEqual(self.ev("-2^2"), -4)
        self.assertEqual(self.ev("(-2)^2"), 4)
        self.assertEqual(self.ev("2^-1"), 0.5)
        self.assertEqual(self.ev("--3"), 3)

    def test_division_keeps_integers_exact(self):
        result = self.ev("6/3")
        self.assertEqual(result, 2)
        self.assertIsInstance(result, int)
        self.assertAlmostEqual(self.ev("1/3"), 1 / 3)

    def test_floor_division_and_modulo(self):
        self.assertEqual(self.ev("7//2"), 3)
        self.assertEqual(self.ev("7%3"), 1)

    def test_factorial(self):
        self.assertEqual(self.ev("5!"), 120)
        self.assertEqual(self.ev("3!!"), 720)
        self.assertEqual(self.ev("-3!"), -6)
        self.assertEqual(self.ev("2^3!"), 64)

    def test_big_integers_are_exact(self):
        self.assertEqual(self.ev("2^100"), 2 ** 100)
        self.assertEqual(self.ev("30!"), math.factorial(30))

    def test_float_noise_is_hidden(self):
        self.assertEqual(self.shown("0.1+0.2"), "0.3")
        self.assertEqual(self.shown("sin(pi)"), "0")

    def test_number_literals(self):
        self.assertEqual(self.ev("0xFF"), 255)
        self.assertEqual(self.ev("0b1010"), 10)
        self.assertEqual(self.ev("0o17"), 15)
        self.assertEqual(self.ev("1e3"), 1000)
        self.assertEqual(self.ev("1_000_000"), 1_000_000)
        self.assertEqual(self.ev(".5"), 0.5)

    def test_unicode_operators(self):
        self.assertEqual(self.ev("6×7"), 42)
        self.assertEqual(self.ev("8÷2"), 4)
        self.assertAlmostEqual(self.ev("2π"), 2 * math.pi)


class TestImplicitMultiplication(CalcTestCase):
    def test_number_and_constant(self):
        self.assertAlmostEqual(self.ev("2pi"), 2 * math.pi)
        self.assertAlmostEqual(self.ev("2e"), 2 * math.e)

    def test_parentheses(self):
        self.assertEqual(self.ev("3(1+2)"), 9)
        self.assertEqual(self.ev("(1+2)(3+4)"), 21)

    def test_variables(self):
        self.calc.execute("x = 4")
        self.assertEqual(self.ev("2x"), 8)
        self.assertEqual(self.ev("x(2)"), 8)


class TestFunctions(CalcTestCase):
    def test_builtins(self):
        self.assertEqual(self.ev("sqrt(16)"), 4)
        self.assertEqual(self.ev("log(100)"), 2)
        self.assertAlmostEqual(self.ev("log(8, 2)"), 3)
        self.assertEqual(self.ev("max(1, 5, 3)"), 5)
        self.assertEqual(self.ev("gcd(12, 18)"), 6)
        self.assertEqual(self.ev("comb(5, 2)"), 10)
        self.assertAlmostEqual(self.ev("root(-27, 3)"), -3)
        self.assertAlmostEqual(self.ev("cbrt(-8)"), -2)

    def test_degree_mode(self):
        self.calc.degrees = True
        self.assertAlmostEqual(self.ev("sin(30)"), 0.5)
        self.assertEqual(self.shown("sin(180)"), "0")
        self.assertEqual(self.shown("tan(45)"), "1")
        self.assertAlmostEqual(self.ev("asin(1)"), 90)
        self.assertCalcError("tan(90)", "не определён")

    def test_wrong_argument_count(self):
        self.assertCalcError("sqrt(1, 2)", "ожидает")
        self.assertCalcError("comb(5)", "ожидает")

    def test_domain_errors(self):
        self.assertCalcError("sqrt(-1)", "области определения")
        self.assertCalcError("ln(0)", "области определения")


class TestVariablesAndUserFunctions(CalcTestCase):
    def test_assignment(self):
        self.assertEqual(self.shown("x = 5"), "x = 5")
        self.assertEqual(self.ev("x*2"), 10)

    def test_ans_and_continuation(self):
        self.ev("2+3")
        self.assertEqual(self.ev("ans*2"), 10)
        self.assertEqual(self.ev("*2"), 20)
        self.assertEqual(self.ev("^2"), 400)

    def test_user_function(self):
        self.calc.execute("f(x) = x^2 + 1")
        self.assertEqual(self.ev("f(3)"), 10)
        self.calc.execute("dist(a, b) = sqrt(a^2 + b^2)")
        self.assertEqual(self.ev("dist(3, 4)"), 5)
        self.assertEqual(self.ev("f(dist(3, 4))"), 26)

    def test_function_reads_globals_at_call_time(self):
        self.calc.execute("k = 2")
        self.calc.execute("h(x) = k*x")
        self.calc.execute("k = 3")
        self.assertEqual(self.ev("h(2)"), 6)

    def test_parameters_shadow_globals(self):
        self.calc.execute("x = 100")
        self.calc.execute("g(x) = x + 1")
        self.assertEqual(self.ev("g(1)"), 2)

    def test_recursion_is_limited(self):
        self.calc.execute("r(x) = r(x)")
        self.assertCalcError("r(1)", "рекурсия")

    def test_reserved_names(self):
        self.assertCalcError("pi = 3", "константа")
        self.assertCalcError("sin = 2", "встроенная")
        self.assertCalcError("ans = 1", "зарезервировано")
        self.assertCalcError("sin(x) = x", "встроенная")


class TestErrors(CalcTestCase):
    def test_division_by_zero(self):
        self.assertCalcError("1/0", "ноль")
        self.assertCalcError("5%0", "ноль")
        self.assertCalcError("0^-1", "Ноль")

    def test_error_position_points_at_problem(self):
        err = self.assertCalcError("2 + foo", "Неизвестная переменная")
        self.assertEqual(err.pos, 4)

    def test_syntax_errors(self):
        self.assertCalcError("(2+3", "скобк")
        self.assertCalcError("2+3)", "скобк")
        self.assertCalcError("2 3", "оператор")
        self.assertCalcError("2+", "оборвалось")
        self.assertCalcError("()", "Пустые")

    def test_eval_is_never_used(self):
        self.assertCalcError("__import__('os').system('echo hacked')")

    def test_huge_results_are_rejected_fast(self):
        self.assertCalcError("9^9^9", "большое")
        self.assertCalcError("10.0^400", "большое")
        self.assertCalcError("5000!", "факториал")

    def test_complex_results_are_rejected(self):
        self.assertCalcError("(-8)^(1/3)", "комплексный")

    def test_deep_nesting_does_not_crash(self):
        self.assertCalcError("(" * 1000 + "1" + ")" * 1000, "вложенность")

    def test_ans_before_any_result(self):
        self.assertCalcError("ans + 1", "ещё нет")


class TestFormatting(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_number(2.0), "2")
        self.assertEqual(format_number(-0.0), "0")
        self.assertEqual(format_number(float('inf')), "∞")
        self.assertEqual(format_number(1 / 3), "0.333333333333")
        self.assertEqual(format_number(10 ** 20), "100000000000000000000")


if __name__ == '__main__':
    unittest.main()
