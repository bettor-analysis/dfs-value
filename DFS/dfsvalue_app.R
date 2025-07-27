library(shiny)
library(readr)
library(dplyr)
library(randomForest)
library(xgboost)
library(Metrics)
library(tibble)
library(reactable)
library(bslib)

# Shiny App: DFS Regression Value Guide with Lumen Theme and Info Tab
ui <- fluidPage(
  theme = bs_theme(bootswatch = "lumen"),
  titlePanel("DFS Regression Value Guide"),
  tabsetPanel(
    tabPanel("Dashboard",
             sidebarLayout(
               sidebarPanel(
                 fileInput("file", "Upload CSV File", accept = c(".csv")),
                 uiOutput("colmap_ui"),
                 actionButton("run", "Run Models", class = "btn btn-primary"),
                 downloadButton("download", "Download Results", class = "btn btn-secondary"),
                 hr(),
                 uiOutput("filters_ui")
               ),
               mainPanel(
                 h3("Model Performance Summary"),
                 reactableOutput("performance"),
                 h3("Value Grades"),
                 reactableOutput("value_table")
               )
             )
    ),
    tabPanel("About",
             h3("How It Works"),
             p("This tool allows you to upload a CSV with three essential columns: Player Name, Projected Points (FPTS), and Salary (SAL). After uploading and mapping your columns, click 'Run Models' to train and compare five different regression models."),
             h4("Models Used"),
             tags$ul(
               tags$li(strong("Linear Regression"), ": Fits a straight-line relationship between Salary and Projected Points."),
               tags$li(strong("Log-Linear Regression"), ": Models the log of Salary against Projected Points to capture exponential-like relationships, then converts predictions back to salary scale."),
               tags$li(strong("Polynomial Regression (Degree 2)"), ": Introduces a squared term of Projected Points to model curvature in the relationship."),
               tags$li(strong("Random Forest"), ": An ensemble of decision trees that captures non-linear patterns and interactions between variables."),
               tags$li(strong("XGBoost"), ": A gradient boosting framework that builds trees sequentially to minimize prediction error, often yielding high accuracy.")
             ),
             h4("Model Selection & Value Grades"),
             p("Each model is evaluated using RMSE, MAE, MAPE, and Adjusted R². The model with the lowest RMSE is selected as the best predictor. Predicted salaries from this model are compared to actual salaries to compute a Value Delta, which is graded as follows:"),
             tags$ul(
               tags$li("A | Great Value: Delta ≥ 600"),
               tags$li("B | Good Value: Delta ≥ 200"),
               tags$li("C | Moderate Value: Delta > -200"),
               tags$li("X | Fade: Delta ≤ -200")
             ),
             p("Use 'Download Results' to export the full dataset with grades for further analysis or sharing."),
             hr(),
             h4("Why It Works"),
             tags$ul(
               tags$li(strong("Salary vs. Points Relationship:"), " DFS salaries generally correlate with projected fantasy points. By modeling salary as a function of FPTS, the app estimates what a player’s salary should be given their projection."),
               tags$li(strong("Value Delta Highlighting:"), " The difference between model-predicted salary and actual salary shows underpriced players (positive delta) and overpriced ones (negative delta)."),
               tags$li(strong("Model Diversity:"), " Combining traditional regression and tree-based machine learning captures both linear trends and complex non-linear relationships in the data.")
             ),
             hr(),
             h4("Limitations to Be Aware Of"),
             tags$ul(
               tags$li(strong("Missing Features:"), " DFS salaries consider many variables beyond projected points (e.g., opponent strength, player popularity, injury risk) that aren’t in this model."),
               tags$li(strong("Projection Quality:"), " The accuracy of value estimates depends on the quality of the input projections. Inaccurate FPTS input leads to biased model outputs."),
               tags$li(strong("Contextual Factors:"), " DFS sites sometimes price players strategically to diversify lineups, so flagged ‘value’ players may carry additional risk or intentional price inefficiencies.")
             )
    )
  )
)

server <- function(input, output, session) {
  # Read uploaded CSV
  df_raw <- reactive({
    req(input$file)
    read_csv(input$file$datapath)
  })
  
  # Dynamic column mapping UI
  output$colmap_ui <- renderUI({
    req(df_raw())
    cols <- colnames(df_raw())
    tagList(
      selectInput("player_col", "Player Column:", choices = cols),
      selectInput("fpts_col", "Projected Points Column (FPTS):", choices = cols),
      selectInput("salary_col", "Salary Column (SAL):", choices = cols)
    )
  })
  
  # Process and rename uploaded data
  df <- reactive({
    req(input$player_col, input$fpts_col, input$salary_col)
    df_raw() %>%
      rename(
        PLAYER = !!sym(input$player_col),
        FPTS   = !!sym(input$fpts_col),
        SAL    = !!sym(input$salary_col)
      ) %>%
      filter(!is.na(FPTS), !is.na(SAL))
  })
  
  # Run models on button click
  results <- eventReactive(input$run, {
    data <- df()
    model_linear <- lm(SAL ~ FPTS, data)
    data$PredLinear <- predict(model_linear, newdata = data)
    model_log    <- lm(log(SAL) ~ FPTS, data)
    data$PredLog <- exp(predict(model_log, newdata = data))
    model_poly   <- lm(SAL ~ poly(FPTS, 2), data)
    data$PredPoly<- predict(model_poly, newdata = data)
    model_rf     <- randomForest(SAL ~ FPTS, data, ntree = 500)
    data$PredRF  <- predict(model_rf, newdata = data)
    X            <- as.matrix(data$FPTS)
    y            <- data$SAL
    model_xgb    <- xgboost(data = X, label = y, nrounds = 100,
                            objective = "reg:squarederror", verbose = 0)
    data$PredXGB <- predict(model_xgb, newdata = X)
    mae_fn  <- function(a, p) mean(abs(a - p))
    mape_fn <- function(a, p) mean(abs((a - p)/a)) * 100
    performance <- tibble(
      Model  = c("Linear","Log-Linear","Polynomial","Random Forest","XGBoost"),
      RMSE   = c(rmse(data$SAL,data$PredLinear),rmse(data$SAL,data$PredLog),rmse(data$SAL,data$PredPoly),rmse(data$SAL,data$PredRF),rmse(data$SAL,data$PredXGB)),
      MAE    = c(mae_fn(data$SAL,data$PredLinear), mae_fn(data$SAL,data$PredLog), mae_fn(data$SAL,data$PredPoly), mae_fn(data$SAL,data$PredRF), mae_fn(data$SAL,data$PredXGB)),
      MAPE   = c(mape_fn(data$SAL,data$PredLinear),mape_fn(data$SAL,data$PredLog),mape_fn(data$SAL,data$PredPoly),mape_fn(data$SAL,data$PredRF),mape_fn(data$SAL,data$PredXGB)),
      R2_Adj = c(summary(model_linear)$adj.r.squared,summary(model_log)$adj.r.squared,summary(model_poly)$adj.r.squared,1- sum((data$SAL-data$PredRF)^2)/sum((data$SAL-mean(data$SAL))^2),1- sum((data$SAL-data$PredXGB)^2)/sum((data$SAL-mean(data$SAL))^2))
    )
    best_model <- performance$Model[which.min(performance$RMSE)]
    table_data <- data %>% mutate(
      PredictedSalary = case_when(
        best_model=="Linear" ~ PredLinear,
        best_model=="Log-Linear" ~ PredLog,
        best_model=="Polynomial" ~ PredPoly,
        best_model=="Random Forest" ~ PredRF,
        best_model=="XGBoost" ~ PredXGB),
      ValueDelta = PredictedSalary - SAL,
      Grade      = case_when(
        ValueDelta>=600 ~ "A | Great Value",
        ValueDelta>=200 ~ "B | Good Value",
        ValueDelta>-200 ~ "C | Moderate Value",
        TRUE            ~ "X | Fade")) %>% arrange(desc(ValueDelta))
    list(performance=performance,table=table_data)
  })
  
  # Filters UI in sidebar
  output$filters_ui <- renderUI({
    req(results())
    tbl <- results()$table
    tagList(
      textInput("player_search","Player Name:",placeholder="Type to search..."),
      sliderInput("salary_range","Salary Range:",min=min(tbl$SAL),max=max(tbl$SAL),value=c(min(tbl$SAL),max(tbl$SAL)),pre="$",step=100),
      selectInput("grade_filter","Grade:",choices=c("All",unique(tbl$Grade)),selected="All")
    )
  })
  
  # Reactive filtered table
  filtered_table <- reactive({
    req(results())
    tbl <- results()$table
    if (!is.null(input$player_search) && nzchar(input$player_search)) tbl <- tbl[grepl(input$player_search,tbl$PLAYER,ignore.case=TRUE),]
    if (!is.null(input$salary_range)) {
      rng <- input$salary_range
      tbl <- tbl[tbl$SAL>=rng[1] & tbl$SAL<=rng[2],]
    }
    if (!is.null(input$grade_filter) && input$grade_filter!="All") tbl <- tbl[tbl$Grade==input$grade_filter,]
    tbl
  })
  
  # Render performance summary
  output$performance <- renderReactable({
    req(results())
    reactable(results()$performance,
              columns=list(
                Model=colDef(align="center"),
                RMSE=colDef(format=colFormat(digits=2),align="center"),
                MAE=colDef(format=colFormat(digits=2),align="center"),
                MAPE=colDef(format=colFormat(digits=1),align="center"),
                R2_Adj=colDef(name="R2 Adj",format=colFormat(digits=3),align="center")
              ),defaultPageSize=5,bordered=TRUE,highlight=TRUE)
  })
  
  # Render filtered value table
  output$value_table <- renderReactable({
    req(filtered_table())
    reactable(filtered_table()%>%select(PLAYER,FPTS,SAL,PredictedSalary,ValueDelta,Grade),
              columns=list(
                PLAYER=colDef(name="Player",align="center"),
                FPTS=colDef(name="Projection",format=colFormat(digits=2),align="center"),
                SAL=colDef(name="Salary",format=colFormat(prefix="$",separators=TRUE),align="center"),
                PredictedSalary=colDef(name="Predicted Salary",format=colFormat(prefix="$",separators=TRUE,digits=2),align="center"),
                ValueDelta=colDef(name="Value Delta",format=colFormat(prefix="$",separators=TRUE,digits=2),align="center"),
                Grade=colDef(name="Grade",align="center")
              ),defaultPageSize=10,bordered=TRUE,highlight=TRUE)
  })
  
  # Download handler
  output$download <- downloadHandler(
    filename=function() paste0("dfs_value_results_",Sys.Date(),".csv"),
    content=function(file) write_csv(results()$table,file)
  )
}

shinyApp(ui=ui,server=server)