import csv
import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path

MOTOR_PATH = Path(__file__).parents[1] / "notebooks" / "06_motor_regras_mvp.py"
spec = importlib.util.spec_from_file_location("motor_regras_mvp", MOTOR_PATH)
motor = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = motor
spec.loader.exec_module(motor)


class TestMotorRegrasMVP(unittest.TestCase):
    def test_caso_h01_totaliza_75_e_faixa_prioritaria(self):
        resultado = motor.avaliar_caso(motor.gerar_caso_h01())
        self.assertEqual(resultado.indice_atencao, 75)
        self.assertEqual(resultado.faixa, "Prioritária")
        self.assertEqual(sum(f["pontos"] for f in resultado.fatores), 75)

    def test_faixas_do_indice(self):
        casos = {
            0: "Rotineira", 39: "Rotineira", 40: "Intermediária",
            69: "Intermediária", 70: "Prioritária", 100: "Prioritária",
        }
        for indice, faixa in casos.items():
            with self.subTest(indice=indice):
                self.assertEqual(motor.classificar_faixa(indice), faixa)

    def test_indice_e_limitado_entre_zero_e_cem(self):
        caso = motor.gerar_caso_h01() | {"impacto_financeiro": "alto"}
        resultado = motor.avaliar_caso(caso)
        self.assertGreaterEqual(resultado.indice_atencao, 0)
        self.assertLessEqual(resultado.indice_atencao, 100)

    def test_registro_hitl_exige_justificativa(self):
        resultado = motor.avaliar_caso(motor.gerar_caso_h01())
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaisesRegex(ValueError, "justificativa"):
                motor.registrar_decisao(resultado, "Aceitar", "   ", "advogado", Path(pasta) / "logs_hitl.csv")

    def test_registro_hitl_persiste_cabecalho_e_dados(self):
        resultado = motor.avaliar_caso(motor.gerar_caso_h01())
        with tempfile.TemporaryDirectory() as pasta:
            log = Path(pasta) / "logs_hitl.csv"
            registro = motor.registrar_decisao(resultado, "Aceitar com ajuste", "Revisar alçada", "Dra. Ana", log)
            with log.open(encoding="utf-8", newline="") as arquivo:
                linhas = list(csv.DictReader(arquivo))
            self.assertEqual(len(linhas), 1)
            self.assertEqual(linhas[0]["caso_id"], "H-01")
            self.assertEqual(linhas[0]["decisao"], "Aceitar com ajuste")
            self.assertEqual(linhas[0]["usuario"], "Dra. Ana")
            self.assertEqual(linhas[0]["versao_motor"], motor.VERSAO_MOTOR)
            self.assertEqual(registro["justificativa"], "Revisar alçada")


if __name__ == "__main__":
    unittest.main(verbosity=2)
