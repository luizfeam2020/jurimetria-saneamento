"""
Motor de regras MVP — índice preliminar de atenção
Framework de Jurimetria para Saneamento Básico

Protótipo acadêmico, transparente e reproduzível. Usa somente regras
ilustrativas e não representa dados ou critérios oficiais da CASAN.

Execução:
    python 06_motor_regras_mvp.py

Integração opcional com Streamlit:
    from importlib.util import spec_from_file_location, module_from_spec
    # importar avaliar_caso, registrar_decisao e gerar_caso_h01

Regras da Tabela 13:
    impacto financeiro: baixo 5, médio 15, alto 30
    tipologia sensível: +20
    fase avançada: +15
    tempo acima do limiar: +10
    histórico semelhante: +0 no MVP
    sinal manual de impacto reputacional: +15
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import csv
import json

VERSAO_MOTOR = "MVP-1.0"
LIMIAR_TEMPO_DIAS = 365

DecisaoHITL = Literal["Aceitar", "Aceitar com ajuste", "Rejeitar"]


@dataclass(frozen=True)
class ConfiguracaoRegras:
    """Parâmetros iniciais e configuráveis do motor."""

    limiar_tempo_dias: int = LIMIAR_TEMPO_DIAS
    pontuacao_impacto_baixo: int = 5
    pontuacao_impacto_medio: int = 15
    pontuacao_impacto_alto: int = 30
    pontuacao_tipologia_sensivel: int = 20
    pontuacao_fase_avancada: int = 15
    pontuacao_tempo_excedido: int = 10
    pontuacao_reputacional: int = 15


@dataclass
class ResultadoAtencao:
    caso_id: str
    indice_atencao: int
    faixa: str
    fatores: list[dict[str, Any]] = field(default_factory=list)
    recomendacao: str = ""
    alerta: str = "Recomendação não substitui o julgamento jurídico."
    qualidade_dados: str = "completa"
    versao_motor: str = VERSAO_MOTOR

    def para_dict(self) -> dict[str, Any]:
        return asdict(self)


def _texto_normalizado(valor: Any) -> str:
    return str(valor or "").strip().lower()


def classificar_faixa(indice: int) -> str:
    if indice < 40:
        return "Rotineira"
    if indice < 70:
        return "Intermediária"
    return "Prioritária"


def _adicionar_fator(fatores: list[dict[str, Any]], nome: str, pontos: int, detalhe: str) -> None:
    if pontos:
        fatores.append({"fator": nome, "pontos": pontos, "detalhe": detalhe})


def avaliar_caso(
    caso: dict[str, Any],
    configuracao: ConfiguracaoRegras | None = None,
) -> ResultadoAtencao:
    """Aplica as regras transparentes e devolve o índice de 0 a 100.

    Campos aceitos: caso_id, valor_causa, impacto_financeiro (baixo/médio/alto),
    tipologia, fase, tempo_tramitacao_dias, impacto_reputacional e
    historico_semelhante. O campo impacto_financeiro, quando ausente, é
    inferido de forma conservadora pelo valor da causa.
    """
    cfg = configuracao or ConfiguracaoRegras()
    caso_id = str(caso.get("caso_id", caso.get("id", "sem-id")))
    fatores: list[dict[str, Any]] = []

    impacto = _texto_normalizado(caso.get("impacto_financeiro"))
    if impacto not in {"baixo", "medio", "médio", "alto"}:
        valor = float(caso.get("valor_causa", 0) or 0)
        impacto = "alto" if valor >= 150_000 else "medio" if valor >= 50_000 else "baixo"
    impacto = "medio" if impacto == "médio" else impacto
    pontos_impacto = {
        "baixo": cfg.pontuacao_impacto_baixo,
        "medio": cfg.pontuacao_impacto_medio,
        "alto": cfg.pontuacao_impacto_alto,
    }[impacto]
    _adicionar_fator(fatores, "Impacto financeiro potencial", pontos_impacto, f"Faixa {impacto}")

    tipologia = _texto_normalizado(caso.get("tipologia", caso.get("assunto_principal")))
    termos_sensiveis = ("consumer", "interrup", "regulat", "tarifa", "abastecimento", "qualidade")
    tipologia_sensivel = bool(caso.get("tipologia_sensivel", False)) or any(t in tipologia for t in termos_sensiveis)
    if tipologia_sensivel:
        _adicionar_fator(fatores, "Tipologia sensível", cfg.pontuacao_tipologia_sensivel, "Demanda com potencial de reiteração ou sensibilidade regulatória")

    fase = _texto_normalizado(caso.get("fase"))
    fases_avancadas = ("instrução final", "instrucao final", "sentença", "sentenca", "recurso", "avançada", "avancada")
    fase_avancada = bool(caso.get("fase_avancada", False)) or any(t in fase for t in fases_avancadas)
    if fase_avancada:
        _adicionar_fator(fatores, "Fase processual avançada", cfg.pontuacao_fase_avancada, "Instrução final, sentença iminente ou recurso")

    tempo = int(caso.get("tempo_tramitacao_dias", caso.get("tempo_total_dias", 0)) or 0)
    if tempo > cfg.limiar_tempo_dias:
        _adicionar_fator(fatores, "Tempo de tramitação", cfg.pontuacao_tempo_excedido, f"{tempo} dias, acima do limiar de {cfg.limiar_tempo_dias}")

    reputacional = bool(caso.get("impacto_reputacional", caso.get("sinal_manual_reputacional", False)))
    if reputacional:
        _adicionar_fator(fatores, "Sinal manual do advogado", cfg.pontuacao_reputacional, "Impacto reputacional marcado no input HITL")

    # O histórico de desfecho semelhante permanece explicitamente sem pontos no MVP.
    indice = min(100, max(0, sum(int(f["pontos"]) for f in fatores)))
    faixa = classificar_faixa(indice)
    recomendacao = (
        "Submeter à gestão e avaliar acordo conforme a alçada interna."
        if faixa == "Prioritária"
        else "Avaliar priorização pela equipe responsável."
        if faixa == "Intermediária"
        else "Manter acompanhamento rotineiro conforme o fluxo interno."
    )
    campos_essenciais = ("caso_id", "valor_causa", "tipologia", "fase", "tempo_tramitacao_dias")
    qualidade = "completa" if all(caso.get(campo) is not None for campo in campos_essenciais) else "parcial — confirmar dados antes de usar"
    return ResultadoAtencao(caso_id, indice, faixa, fatores, recomendacao, qualidade_dados=qualidade)


def gerar_caso_h01() -> dict[str, Any]:
    """Retorna o caso hipotético da tese, que deve resultar em índice 75."""
    return {
        "caso_id": "H-01",
        "tipologia": "Demanda consumerista por falha no abastecimento",
        "tipologia_sensivel": True,
        "comarca": "Comarca hipotética",
        "valor_causa": 80_000.00,
        "impacto_financeiro": "medio",
        "fase": "avançada",
        "fase_avancada": True,
        "tempo_tramitacao_dias": LIMIAR_TEMPO_DIAS + 1,
        "impacto_reputacional": True,
    }


def registrar_decisao(
    resultado: ResultadoAtencao,
    decisao: DecisaoHITL,
    justificativa: str,
    usuario: str = "nao-informado",
    caminho_log: str | Path = "logs_hitl.csv",
) -> dict[str, Any]:
    """Registra a decisão humana com data/hora, usuário e versão do motor."""
    if decisao not in {"Aceitar", "Aceitar com ajuste", "Rejeitar"}:
        raise ValueError("Decisão HITL inválida.")
    if not str(justificativa).strip():
        raise ValueError("A justificativa é obrigatória para registrar a decisão.")
    registro = {
        "data_hora_utc": datetime.now(timezone.utc).isoformat(),
        "usuario": usuario,
        "caso_id": resultado.caso_id,
        "indice_atencao": resultado.indice_atencao,
        "faixa": resultado.faixa,
        "decisao": decisao,
        "justificativa": str(justificativa).strip(),
        "versao_motor": resultado.versao_motor,
    }
    destino = Path(caminho_log)
    destino.parent.mkdir(parents=True, exist_ok=True)
    novo_arquivo = not destino.exists()
    with destino.open("a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(registro))
        if novo_arquivo:
            escritor.writeheader()
        escritor.writerow(registro)
    return registro


def imprimir_resultado(resultado: ResultadoAtencao) -> None:
    print(f"Caso: {resultado.caso_id}")
    print(f"Índice de atenção: {resultado.indice_atencao}/100")
    print(f"Faixa: {resultado.faixa}")
    print("Fatores explicativos:")
    for fator in resultado.fatores:
        print(f"  - {fator['fator']}: +{fator['pontos']} ({fator['detalhe']})")
    print(f"Recomendação: {resultado.recomendacao}")
    print(f"Qualidade dos dados: {resultado.qualidade_dados}")
    print(f"Alerta: {resultado.alerta}")


def main() -> None:
    resultado = avaliar_caso(gerar_caso_h01())
    imprimir_resultado(resultado)
    assert resultado.indice_atencao == 75, "O caso H-01 deve totalizar 75 pontos."
    print("\nValidação do caso H-01: OK")
    print("Nenhuma decisão HITL foi registrada automaticamente; a decisão deve ser preenchida por advogado.")
    print("\nJSON do resultado:")
    print(json.dumps(resultado.para_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
