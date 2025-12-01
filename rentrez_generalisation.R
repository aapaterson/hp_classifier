library(magrittr)

# rentrez stuff
query <- "ecology*[Title/Abstract]
    NOT (infectious [Title/Abstract])
    NOT (disease [Title/Abstract])
    NOT (diseases [Title/Abstract])
    NOT (virus [Title/Abstract])
    NOT (host [Title/Abstract])
    NOT (hosts [Title/Abstract])
    NOT (pathogen [Title/Abstract])
    NOT (parasite [Title/Abstract])
    AND english[LA]
    AND Journal Article[PT]
    AND 2024:2025[DP]"

ids <- rentrez::entrez_search(
    db = "pubmed", term = query, retmax = 100, rettype = "xml"
)$ids
record <- xml2::read_xml(rentrez::entrez_fetch(db = "pubmed", id = ids, rettype = "xml"))
dataframe_abs <- xml2::xml_find_all(record, "//AbstractText")
dataframe_ids <- xml2::xml_text(xml2::xml_find_all(record, "//PMID"))
dataframe_abs

extract_abs <- function(id) {
    record <- xml2::read_xml(rentrez::entrez_fetch(db = "pubmed", id = id, rettype = "xml"))
    dataframe_abs <- paste(xml2::xml_text(xml2::xml_find_all(record, "//AbstractText")), collapse = ", ")
    dataframe_abs
}

df <- data.frame(ids = dataframe_ids) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(abs = extract_abs(ids)) %>%
    tibble::tibble()

tail(df)

## current
bibliometrix::convert2df(dbsource = "pubmed", format = "api")

df_p <- pubmedR::pmApiRequest(query = query, limit = 100, api_key = NULL) %>%
    bibliometrix::convert2df(
        dbsource = "pubmed", format = "api"
    )
colnames(df_p)
df_p$PMID %in% dataframe_ids
df_p$UT

entrezCorpus <- function(term, db, n) {
    extract_abs <- function(id) {
        record <- xml2::read_xml(rentrez::entrez_fetch(db = "pubmed", id = id, rettype = "xml"))
        dataframe_abs <- paste(xml2::xml_text(xml2::xml_find_all(record, "//AbstractText")), collapse = ", ")
        dataframe_abs
    }

    ids <- rentrez::entrez_search(
        db = db, term = query, retmax = 100, rettype = "xml"
    )$ids
    record <- xml2::read_xml(rentrez::entrez_fetch(db = db, id = ids, rettype = "xml"))
    dataframe_abs <- xml2::xml_find_all(record, "//AbstractText")


    df <- data.frame(ids = ids) %>%
        dplyr::mutate(abs = sapply(ids, extract_abs)) %>%
        tibble::tibble()
}

df <- entrezCorpus(
    db = "pubmed", term = query, n = 100
)
df



ids <- rentrez::entrez_search(
    db = "snp", term = "Tetrahymena thermophila[ORGN] AND 2013:2015[PDAT]", retmax = 100, rettype = "xml"
)$ids
ids
record <- xml2::read_xml(rentrez::entrez_fetch(db = "snp", id = ids, rettype = "xml"))
dataframe_abs <- xml2::xml_find_all(record, "//AbstractText")
dataframe_abs

vapply()

extract_abs <- function(id) {
    record <- xml2::read_xml(rentrez::entrez_fetch(db = "pubmed", id = id, rettype = "xml"))
    dataframe_abs <- paste(xml2::xml_text(xml2::xml_find_all(record, "//AbstractText")), collapse = ", ")
    dataframe_abs
}

ids <- rentrez::entrez_search(
    db = db, term = query, retmax = 100, rettype = "xml"
)$ids
record <- xml2::read_xml(rentrez::entrez_fetch(db = db, id = ids, rettype = "xml"))
dataframe_abs <- xml2::xml_find_all(xml2::as_list(record), "//AbstractText")

ids <- rentrez::entrez_search(
    db = "pubmed", term = query, retmax = 100, rettype = "xml"
)$ids
record <- xml2::read_xml(rentrez::entrez_fetch(db = "pubmed", id = ids, rettype = "xml"))
record
dataframe_abs <- paste(xml2::xml_text(xml2::xml_find_all(record, "//AbstractText")), collapse = ", ")
dataframe_abs
ids

record <- xml2::xml_ns_strip(record)
xml2::xml_find_all(record, "//AbstractText", flatten = TRUE)[99:103]
xml2::xml_find_all(record, "//AbstractText", flatten = TRUE)[99:103]
record
xml2::xml_find_all(record, "//AbstractText", flatten = FALSE)

xml2::xml_ns(record)

tibble::tibble(xml2::as_list(record)

df <- data.frame(ids = ids) %>%
    dplyr::mutate(abs = sapply(ids, extract_abs)) %>%
    tibble::tibble()

sapply(ids, extract_abs)

bibliometrix::convert2df(
    file = record, dbsource = "pubmed", format = "plaintext"
)

dplyr::select(abs = AB, absID = PMID) %>%
tidyr::tibble()
