numeric_columns <- c("Age", "Fare", "FamilySize")
category_levels <- list(Pclass = c("1", "2", "3"), Sex = c("female", "male"),
                        Embarked = c("C", "Q", "S"))
required_columns <- c("PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")

validate <- function(data, name, labeled = FALSE) {
  required <- c(required_columns, if (labeled) "Survived")
  missing <- setdiff(required, names(data))
  if (length(missing)) stop(paste(name, "missing columns:", paste(missing, collapse = ", ")))
  if (!nrow(data)) stop(paste(name, "has no passenger rows"))
  ids <- data$PassengerId
  if (!is.numeric(ids) || anyNA(ids) || any(!is.finite(ids)) || anyDuplicated(ids) ||
      any(ids < 1 | ids > 1000000 | ids %% 1 != 0)) {
    stop(paste(name, "PassengerId must be unique integers from 1 to 1000000"))
  }
  for (column in c("Age", "Fare", "SibSp", "Parch")) {
    values <- data[[column]][!is.na(data[[column]])]
    if (length(values) && (!is.numeric(values) || any(!is.finite(values)) || any(values < 0))) {
      stop(paste(name, column, "must be nonnegative finite numbers or missing"))
    }
  }
  for (column in names(category_levels)) {
    values <- as.character(data[[column]][!is.na(data[[column]])])
    if (!all(values %in% category_levels[[column]])) stop(paste(name, "invalid", column, "value"))
  }
  if (labeled) {
    if (anyNA(data$Survived) || !all(data$Survived %in% c(0, 1))) {
      stop(paste(name, "Survived must contain only 0 or 1"))
    }
    counts <- table(factor(data$Survived, levels = c(0, 1)))
    if (any(counts < 2)) stop(paste(name, "needs at least two passengers in each outcome class"))
  }
}

features <- function(data) {
  result <- data[c("Age", "Fare", "Pclass", "Sex", "Embarked")]
  result$FamilySize <- data$SibSp + data$Parch + 1
  result
}

validation_mask <- function(data) {
  # Same stratified ID-based split as Python; double arithmetic avoids integer overflow.
  score <- (as.double(data$PassengerId) * 1103515245 + 42) %% 2147483648
  mask <- rep(FALSE, nrow(data))
  for (outcome in c(0, 1)) {
    indices <- which(data$Survived == outcome)
    ordered <- indices[order(score[indices], data$PassengerId[indices])]
    count <- max(1, floor(length(indices) * 0.2))
    mask[head(ordered, count)] <- TRUE
  }
  mask
}

learn_preprocessing <- function(data) {
  if (any(vapply(data, function(x) all(is.na(x)), logical(1)))) {
    stop("A feature is entirely missing in the fitting subset")
  }
  medians <- vapply(data[numeric_columns], stats::median, numeric(1), na.rm = TRUE)
  modes <- vapply(data[names(category_levels)], function(x) {
    counts <- table(as.character(x))
    names(counts)[which.max(counts)]
  }, character(1))
  list(medians = medians, modes = modes)
}

apply_preprocessing <- function(data, parameters) {
  for (column in numeric_columns) {
    data[[column]][is.na(data[[column]])] <- parameters$medians[[column]]
  }
  for (column in names(category_levels)) {
    values <- as.character(data[[column]])
    values[is.na(values)] <- parameters$modes[[column]]
    data[[column]] <- factor(values, levels = category_levels[[column]])
  }
  data
}

describe_preprocessing <- function(parameters) {
  cat("Numeric medians:\n")
  print(parameters$medians)
  cat("Categorical modes:\n")
  print(parameters$modes)
  cat("Categorical factors use treatment contrasts; numeric features stay in original units.\n")
}

fit_model <- function(data, outcome) {
  data$Survived <- outcome
  model <- stats::glm(Survived ~ Age + Fare + FamilySize + Pclass + Sex + Embarked,
                      data = data, family = stats::binomial(), control = stats::glm.control(maxit = 100))
  if (!model$converged) stop("Logistic regression did not converge")
  model
}

classify <- function(model, data) {
  probabilities <- stats::predict(model, newdata = data, type = "response")
  if (anyNA(probabilities) || any(!is.finite(probabilities))) stop("Invalid model predictions")
  as.integer(probabilities >= 0.5)
}

run <- function(data_dir = "src/data", output_dir = "results/r") {
  for (name in c("train.csv", "test.csv")) {
    path <- file.path(data_dir, name)
    if (!file.exists(path)) stop(paste("Missing", path, "- download the Kaggle files; see README."))
  }
  train <- utils::read.csv(file.path(data_dir, "train.csv"), na.strings = c("", "NA"), stringsAsFactors = FALSE)
  test <- utils::read.csv(file.path(data_dir, "test.csv"), na.strings = c("", "NA"), stringsAsFactors = FALSE)
  validate(train, "train.csv", labeled = TRUE)
  validate(test, "test.csv")
  if (length(intersect(train$PassengerId, test$PassengerId))) {
    stop("train.csv and test.csv contain overlapping PassengerId values")
  }
  cat(sprintf("Loaded %d training passengers and %d test passengers.\n", nrow(train), nrow(test)))
  cat("Training missing values:\n")
  print(colSums(is.na(train[required_columns])))
  cat("Test missing values:\n")
  print(colSums(is.na(test[required_columns])))
  cat(sprintf("Overall training survival rate: %.4f\n", mean(train$Survived)))
  for (column in c("Sex", "Pclass")) {
    cat("Training survival rate by", column, ":\n")
    print(tapply(train$Survived, train[[column]], mean))
  }
  cat("Created FamilySize = SibSp + Parch + 1. Missing components yield missing FamilySize.\n")
  cat("Features:", paste(c(numeric_columns, names(category_levels)), collapse = ", "), "\n")
  cat("PassengerId is used only for splitting/export. Name, Ticket, and Cabin are excluded.\n")
  x <- features(train)
  mask <- validation_mask(train)
  fit_y <- train$Survived[!mask]
  val_y <- train$Survived[mask]
  parameters <- learn_preprocessing(x[!mask, ])
  fit_x <- apply_preprocessing(x[!mask, ], parameters)
  val_x <- apply_preprocessing(x[mask, ], parameters)
  cat(sprintf("Stratified holdout: %d fitting rows, %d validation rows.\n", nrow(fit_x), nrow(val_x)))
  cat("Preprocessing learned only from the fitting subset:\n")
  describe_preprocessing(parameters)
  model <- fit_model(fit_x, fit_y)
  fit_accuracy <- mean(classify(model, fit_x) == fit_y)
  val_predictions <- classify(model, val_x)
  val_accuracy <- mean(val_predictions == val_y)
  majority <- as.integer(names(which.max(table(fit_y))))
  baseline <- mean(val_y == majority)
  cat(sprintf("Fitting-subset accuracy: %.4f\n", fit_accuracy))
  cat(sprintf("Holdout validation accuracy: %.4f\n", val_accuracy))
  cat(sprintf("Majority-class holdout baseline: %.4f\n", baseline))
  cat("Validation confusion matrix (rows actual 0/1, columns predicted 0/1):\n")
  print(table(actual = factor(val_y, levels = c(0, 1)), predicted = factor(val_predictions, levels = c(0, 1))))
  parameters <- learn_preprocessing(x)
  full_x <- apply_preprocessing(x, parameters)
  final <- fit_model(full_x, train$Survived)
  cat("Refitted preprocessing and logistic regression on all labeled training rows:\n")
  describe_preprocessing(parameters)
  full_accuracy <- mean(classify(final, full_x) == train$Survived)
  cat(sprintf("Full-training accuracy (in-sample): %.4f\n", full_accuracy))
  predictions <- classify(final, apply_preprocessing(features(test), parameters))
  cat("Official test accuracy: unavailable; Kaggle test.csv has no true Survived labels.\n")
  cat("gender_submission.csv contains example predictions, not ground truth.\n")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(data.frame(PassengerId = test$PassengerId, Survived = predictions),
                   file.path(output_dir, "submission.csv"), row.names = FALSE)
  metrics <- data.frame(fit_rows = nrow(fit_x), validation_rows = nrow(val_x),
                        train_rows = nrow(train), test_rows = nrow(test),
                        fit_accuracy = fit_accuracy, validation_accuracy = val_accuracy,
                        baseline_accuracy = baseline, full_train_accuracy = full_accuracy)
  utils::write.csv(metrics, file.path(output_dir, "metrics.csv"), row.names = FALSE)
  cat("Saved", length(predictions), "predictions to", file.path(output_dir, "submission.csv"), "\n")
  cat("Saved metrics to", file.path(output_dir, "metrics.csv"), ". No passenger rows dropped.\n")
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 2L) stop("Usage: Rscript src/r/train.R [data_directory] [output_directory]")
  data_dir <- if (length(args) >= 1L) args[[1]] else "src/data"
  output_dir <- if (length(args) >= 2L) args[[2]] else "results/r"
  tryCatch(run(data_dir, output_dir), error = function(error) {
    message("Error: ", conditionMessage(error))
    quit(save = "no", status = 1L)
  })
}
