# All dependencies ship with R; no CRAN download is needed.
for (package in c("stats", "utils", "tools")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(paste("Missing R distribution package:", package))
  }
  cat(package, "version", as.character(utils::packageVersion(package)), "\n")
}
