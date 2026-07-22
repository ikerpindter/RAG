from src.config import FILINGS


def test_cuatro_filings_con_campos_completos():
    assert len(FILINGS) == 4
    required = {"doc_id", "company", "fiscal_year", "url", "filename"}
    for doc in FILINGS:
        assert required <= set(doc), f"faltan campos en {doc.get('doc_id')}"
        assert doc["url"].startswith("https://www.sec.gov/")


def test_doc_ids_unicos_y_catalogo_esperado():
    doc_ids = [doc["doc_id"] for doc in FILINGS]
    assert len(set(doc_ids)) == 4
    assert {doc["company"] for doc in FILINGS} == {"Lennar", "D.R. Horton"}
    assert {doc["fiscal_year"] for doc in FILINGS} == {2023, 2024}
