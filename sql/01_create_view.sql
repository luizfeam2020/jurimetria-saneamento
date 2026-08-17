CREATE OR REPLACE VIEW VW_JURIMETRIA_SANEAMENTO AS
SELECT
    p.id_processo,
    p.numero_unico,
    p.classe,
    p.assunto_principal,
    p.assunto_secundario,
    p.valor_causa,
    p.data_distribuicao,
    p.data_primeira_decisao,
    p.data_sentenca,
    p.data_transito_julgado,
    p.tribunal,
    p.vara,
    p.comarca,
    p.tipo_parte_autora,
    p.tipo_parte_reu,
    p.resultado_sentenca,
    p.resultado_recurso,
    p.valor_condenacao,
    p.valor_acordo,
    p.custas_processuais,
    p.honorarios_sucumbencia,
    p.tempo_total_dias,
    p.numero_audiencias,
    p.numero_pericias,
    p.numero_recursos,
    p.assunto_saneamento,
    p.tipo_demanda,
    p.regiao_geografica,
    p.ano_distribuicao,
    p.mes_distribuicao,
    ROUND(p.valor_condenacao / NULLIF(p.valor_causa, 0), 4) AS taxa_sucumbencia,
    CASE 
        WHEN p.resultado_sentenca = 'PROCEDENTE' THEN 1
        WHEN p.resultado_sentenca = 'IMPROCEDENTE' THEN 0
        ELSE 0.5
    END AS target_procedencia,
    CASE WHEN p.valor_acordo > 0 THEN 1 ELSE 0 END AS flag_acordo,
    ROUND((p.data_sentenca - p.data_distribuicao), 0) AS dias_ate_sentenca,
    ROUND((p.data_transito_julgado - p.data_distribuicao), 0) AS dias_total_processo,
    CASE 
        WHEN p.polo_autor LIKE '%CONSUMIDOR%' THEN 'CIDADAO_ANONIMIZADO_' || p.id_processo
        ELSE p.polo_autor
    END AS polo_autor_anonimizado,
    CASE 
        WHEN p.polo_reu LIKE '%CONSUMIDOR%' THEN 'CIDADAO_ANONIMIZADO_' || p.id_processo
        ELSE p.polo_reu
    END AS polo_reu_anonimizado
FROM LEGARO.PROCESSOS p
WHERE p.assunto_saneamento IS NOT NULL
  AND p.data_distribuicao >= ADD_MONTHS(SYSDATE, -120);
