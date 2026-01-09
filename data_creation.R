set.seed(4)
library(magrittr)

# Create dataset with tidied corpus

pubmedQuery <- function(query, n = 1000) {
    AB <- PMID <- API_KEY <- NULL
    tryCatch(
        {
            count <- pubmedR::pmQueryTotalCount(
                query = query,
                api_key = API_KEY
            )$total_count
            if (count <= 1) {
                return(NULL)
            }
            n <- min(n, count)
            res <- pubmedR::pmApiRequest(
                query = query,
                limit = n,
                api_key = API_KEY
            )
            bibliometrix::convert2df(
                res,
                dbsource = "pubmed",
                format = "api"
            ) %>%
                dplyr::select(abs = AB, absID = PMID)
        },
        error = function(e) {
            NULL
        }
    )
}

intervals <- function(start, end, n) {
    step <- (end - start + 1) / n # add one to inlcude the start point
    starts <- seq(start, end, step) # interval start dates
    ends <- seq(start, end + 1, step)[-1] - 1 # paired end dates ^
    paste("AND ", starts, ":", ends, "[DP]", sep = "") # return search term
}

psuedo_absence <- function(queries, dates, sample_size, n_intervals) {
    fetch_query <- function(query) {
        date_prompts <- intervals(dates[1], dates[2], n_intervals)
        queries <- paste(query, date_prompts)
        interval_sample <- sample_size / n_intervals
        dplyr::bind_rows(lapply(queries, pubmedQuery, n = interval_sample))
    }

    dplyr::bind_rows(lapply(queries, fetch_query))
}

corpus_processing <- function(df) {
    df %>%
        tm::VectorSource() %>%
        tm::VCorpus() %>%
        tm::tm_map(tm::content_transformer(tolower)) %>%
        tm::tm_map(tm::removePunctuation) %>%
        tm::tm_map(tm::removeNumbers) %>%
        tidytext::tidy() %>%
        dplyr::pull(text)
}

training_data <- function(queries, dates, sample_size, n_intervals) {
    pmid <- absID <- NULL
    id_data <- vroom::vroom(
        "../../hp-extractor/raw_data/hp_metadata.tsv",
        delim = "\t",
        show_col_types = FALSE
    ) %>%
        dplyr::select(absID, pmid) %>%
        dplyr::distinct()

    psuedo_absence_data <- psuedo_absence(
        queries = queries,
        dates = dates,
        sample_size = sample_size,
        n_intervals = n_intervals
    ) %>%
        tibble::add_column(response = 0) %>%
        dplyr::filter((absID %in% id_data$pmid) == FALSE) %>%
        dplyr::distinct()

    data <- vroom::vroom(
        "../../hp-extractor/raw_data/hp_abstracts.tsv",
        delim = "\t",
        show_col_types = FALSE
    ) %>%
        dplyr::inner_join(id_data) %>%
        dplyr::mutate(absID = as.character(pmid)) %>%
        dplyr::select(!pmid) %>%
        tibble::add_column(response = 1) %>%
        dplyr::filter(is.na(absID) == FALSE) %>%
        dplyr::bind_rows(psuedo_absence_data) %>%
        dplyr::mutate(abs = gsub("\n", "", abs)) %>%
        dplyr::mutate(abs = gsub("\t", " ", abs)) %>%
        dplyr::mutate(abs = corpus_processing(abs)) %>%
        dplyr::filter(abs != "na")

    # data
    readr::write_tsv(data, "abs_data.tsv")
}

generate_negative_queries <- function() {
    Species <- genus_name <- sp_epithet <- NULL

    virus_names <- vroom::vroom(
        "../../data/virus.csv",
        delim = ",",
        show_col_types = FALSE
    ) %>%
        dplyr::select(Species)

    bacteria_names <- vroom::vroom(
        "../../data/bacteria.csv",
        delim = ",",
        show_col_types = FALSE
    ) %>%
        dplyr::mutate(Species = paste(genus_name, sp_epithet)) %>%
        dplyr::filter(!grepl("NA", Species)) %>%
        dplyr::select(Species) %>%
        dplyr::distinct()

    query <- paste(
        "[Text Word]",
        "AND",
        "('infection'[Text Word]",
        "OR 'disease'[Text Word]",
        "OR 'human'[Text Word]",
        "OR 'animal'[Text Word]",
        "OR 'zoo'[Text Word]",
        "OR 'vet'[Text Word]",
        "OR 'epidemic'[Text Word]",
        "OR 'epizootic'[Text Word])"
    )

    queries_virus <- paste(virus_names$Species, query)
    queries_bacteria <- paste(bacteria_names$Species, query)

    queries <- c(queries_virus, queries_bacteria)
    queries
    # sample(queries, ceiling(length(queries)))
}

generate_training_data <- function() {
    queries <- generate_negative_queries()

    training_data(
        queries = queries,
        dates = c(1961, 2016),
        sample_size = 200,
        n_intervals = 1
    )
}

generate_training_data()
