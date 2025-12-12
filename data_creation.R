library(magrittr)

# Create dataset with tidied corpus

pubmedQuery <- function(query, n = 1000) {
    AB <- PMID <- NULL
    data <- pubmedR::pmApiRequest(query = query, limit = n, api_key = NULL) %>%
        bibliometrix::convert2df(
            dbsource = "pubmed", format = "api"
        ) %>%
        dplyr::select(abs = AB, absID = PMID) %>%
        tidyr::tibble()
    data
}

intervals <- function(start, end, n) {
    step <- (end - start + 1) / n
    starts <- seq(start, end, step)
    ends <- seq(start, end + 1, step)[-1] - 1
    paste("AND ", starts, ":", ends, "[DP]", sep = "")
}

psuedo_absence <- function(query, dates, sample_size, n_intervals) {
    date_prompts <- intervals(dates[1], dates[2], n_intervals)
    queries <- paste(query, date_prompts)
    interval_sample <- sample_size / n_intervals
    dplyr::bind_rows(lapply(queries, pubmedQuery, n = interval_sample))
}

fresh_data <- function(query, n = 1000) {
    new_data <- pubmedQuery(
        query <- query,
        n = n
    ) %>%
        dplyr::mutate(abs = corpus_creation(abs))

    new_data
}

corpus_creation <- function(df) {
    df %>%
        tm::VectorSource() %>%
        tm::VCorpus() %>%
        tm::tm_map(tm::content_transformer(tolower)) %>%
        tm::tm_map(tm::removePunctuation) %>%
        tm::tm_map(tm::removeNumbers) %>%
        tm::tm_map(tm::removeWords, tm::stopwords("english")) %>%
        tm::tm_map(tm::stemDocument) %>%
        broom::tidy() %>%
        dplyr::pull(text)
}


training_data <- function(queries, dates, sample_size, n_intervals) {
    pmid <- absID <- NULL
    id_data <- vroom::vroom(
        "../../hp-extractor/raw_data/hp_metadata.tsv",
        delim = "\t"
    ) %>%
        dplyr::select(absID, pmid) %>%
        dplyr::distinct()

    psuedo_absence_data <- dplyr::bind_rows(
        lapply(queries,
            psuedo_absence,
            dates = dates,
            sample_size = sample_size,
            n_intervals = n_intervals
        )
    ) %>%
        tibble::add_column(response = 0) %>%
        dplyr::distinct()

    data <- vroom::vroom(
        "../../hp-extractor/raw_data/hp_abstracts.tsv",
        delim = "\t"
    ) %>%
        dplyr::inner_join(id_data) %>%
        dplyr::mutate(absID = as.character(pmid)) %>%
        dplyr::select(!pmid) %>%
        tibble::add_column(response = 1) %>%
        dplyr::bind_rows(psuedo_absence_data) %>%
        dplyr::mutate(abs = corpus_creation(abs)) %>%
        dplyr::filter(abs != "na")

    readr::write_tsv(data, "abs_data.tsv")
}

generate_negative_queries <- function() {
    Species <- genus_name <- sp_epithet <- NULL

    virus_names <- vroom::vroom(
        "~/Downloads/virus.csv",
        delim = ","
    ) %>%
        dplyr::select(Species)

    bacteria_names <- vroom::vroom(
        "~/Downloads/lpsn_gss_2025-12-12.csv",
        delim = ","
    ) %>%
        dplyr::mutate(bact = paste(genus_name, sp_epithet)) %>%
        dplyr::filter(!grepl("NA", Species)) %>%
        dplyr::select(Species) %>%
        dplyr::distinct()

    query <- "[Text Word]
        AND 'infection'
		OR [Text Word] 'disease'
		OR [Text Word] 'human'
		OR [Text Word] 'animal'
		OR [Text Word] 'zoo'
		OR [Text Word] 'vet'
		OR [Text Word] 'epidemic'
        OR 'epizootic' [Text Word]
        AND 1960:2016[DP]"

    queries_virus <- paste(virus_names$Species, query)
    queries_bacteria <- paste(bacteria_names$bact, query)

    queries <- c(queries_virus, queries_bacteria)
    sample(queries, ceiling(length(queries) / 1000))
}

generate_training_data <- function() {
    queries <- generate_negative_queries()

    training_data(
        queries = queries,
        dates = c(1961, 2016),
        sample_size = 210,
        n_intervals = 7
    )
}

# generate_training_data() # missing 2 entries for some reason !!!!!

main <- function() {
    today <- format(Sys.Date(), "%Y/%m/%d")
    yesterday <- format(Sys.Date() - 1, "%Y/%m/%d")
    yesterday
    today

    new_df <- fresh_data(
        query = glue::glue("ecology*[Title/Abstract]
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
    AND {yesterday}:{today}[DP]"),
        n = 250
    )

    readr::write_tsv(new_df, "new_data.tsv")
}

# main()
