# Wikipedia data

LwLL is offering wikipedia data for monolingual corpora as part of the machine translation task. The following information are the instructions of how these corpora were collected and processed. For obtaining copies of the post processed corpora, see the main `README.md` for instructions on downloading and uncompressing via the `download.py` utility.

# Code

The code for this processing resides here: https://github.com/tballison/hodgepodge/tree/master/wiki-munging

The commit used for this processing was: `5c1d984`

Wikimedia publishes dumps of "pages and articles" from Wikipedia by language at fairly regular intervals.

These files are bz2 compressed files with heavy wiki-markup.

There's xml markup for article boundaries, but then
wikimedia's own special markup for links, tables and other formatting.

In addition to scraping text out of the wikimedia markup, one also has to filter out non-article content,
such as redirects and other non-article namespaces (see https://en.wikipedia.org/wiki/Wikipedia:Namespace).

The output of this process is a large table with a single sentence per line. 
The text is UTF-8 encoded; all [\n\r\t] have been replaced by ' ' within a sentence. 
We used Stanford's corenlp module to perform sentence segmentation for each article.
At the performer's request, we added some regular expressions to remove list markup (e.g. line initial `*` and bullets).
We also removed duplicate sentences. Finally, the table has been compressed with `.gz`.

# Steps

## Requirements

1. git
1. maven
1. java jdk 11

## Build the project

1. `git clone https://github.com/tballison/hodgepodge`
1. `cd hodgepodge`
1. (Optional) `git checkout 5c1d984
`
1. `cd wiki-munging`
1. `mvn clean install`

## Run the code

The jar file `wiki-munging-0.1-SNAPSHOT.jar` is in `target/` after the build

1. Move the jar file to a working directory
1. Get the wikipedia dumps: `java -cp wiki-munging-0.1-SNAPSHOT.jar GetWikiPageArticleBzips <lang> <bzip_dir>` as in
   `java -cp wiki-munging-0.1-SNAPSHOT.jar GetWikiPageArticleBzips en bzips_en`
1. Run the code that extracts content from the wikipedia marked up articles.: `java -cp wiki-munging-0.1-SNAPSHOT.jar WikiToTableSimple -i <bzip_dir> -o <table_file.gz> -l <language>` as in
   `java -cp wiki-munging-0.1-SNAPSHOT.jar WikiToTableSimple -i bzips_en -o en_table.gz -l en`

## What the code does

1. `GetWikiPageArticleBzips` goes to https://dumps.wikimedia.org/<lang>wiki/latest (
   e.g. https://dumps.wikimedia.org/<lang>wiki/latest) and scrapes all the `latest-pages-articles-*.bz2`
   urls and then retrieves those urls. For some languages, there is only one `.bz2` file; for others,
   there can be many more.

2. `WikiToTableSimple` relies on Jimmy Lin's wikiclean library (https://github.com/lintool/wikiclean) to
   iterate through the articles in the `.bz2` files, select the non-redirect "articles", and remove all of the
   wiki markup.  Further, this code uses Stanford's corenlp module to perform sentence segmentation.
   We also remove list markup with custom regexes and perform deduplication on sentences.  This
   code is single threaded and stores a hash for every sentence in memory.  I was able to get
   it to work on the full English wikipedia dump with `-Xmx20g`.
   
## Sidenote
I wrote code to write sentences to a Postgresql database.  This allowed for multithreading,
and the loading was far faster than the single-threaded `WikiToTableSimple` code, but it crashed
the Postgresql server on my laptop when I tried to run it against the full English wikipedia.

## For posterity
`aws --profile saml-pub s3 cp wiki-en-table-20200716.gz s3://lwll-datasets/compressed_datasets/external/machine_translation/`