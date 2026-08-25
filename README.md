# Jurimetria aplicada ao saneamento básico

Framework demonstrativo de jurimetria e análise preditiva para contencioso em saneamento básico.

> **Aviso:** o projeto usa dados sintéticos e tem finalidade acadêmica/demonstrativa. Nenhum resultado constitui decisão jurídica, recomendação automática ou substitui análise profissional, contraditório e revisão humana.

## Como executar

Na raiz do repositório:

```bash
python -m pip install -r requirements.txt
streamlit run notebooks/05_dashboard_streamlit.py
```

No Windows, também é possível usar:

```bat
streamlit run notebooks\\05_dashboard_streamlit.py
```

Para testar o motor de regras isoladamente:

```bash
python notebooks/06_motor_regras_mvp.py
```

## Módulos

- `notebooks/01_exploracao_e_jurimetria.py`: exploração e indicadores descritivos.
- `notebooks/02_modelagem_classificacao.py`: classificação preditiva.
- `notebooks/03_modelagem_regressao.py`: modelagem de regressão.
- `notebooks/04_explicabilidade_modelos.py`: explicabilidade dos modelos.
- `notebooks/05_dashboard_streamlit.py`: dashboard interativo com cinco abas, dados sintéticos, predição e integração com o motor de regras.
- `notebooks/06_motor_regras_mvp.py`: índice de atenção transparente, fatores explicativos, recomendação orientativa e registro HITL.

## Integração do motor de regras

A aba **Predição e HITL** calcula o índice de atenção do caso selecionado, exibe faixa, fatores e recomendação, e permite registrar uma decisão humana com usuário e justificativa obrigatória. Os registros são gravados em `logs_hitl.csv` durante a execução local.

As faixas do MVP são:

- 0–39: Rotineira;
- 40–69: Intermediária;
- 70–100: Prioritária.

## Uso responsável

O índice é preliminar, explicável e configurável, mas não foi validado em processos reais. Não deve ser usado para automatizar decisões, negar direitos, substituir advogado ou dispensar revisão institucional. Antes de qualquer uso aplicado, é necessário validar dados, critérios, governança, segurança, vieses e conformidade jurídica.
