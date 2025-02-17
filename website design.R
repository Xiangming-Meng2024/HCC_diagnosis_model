required_packages <- c("shiny", "randomForest", "readxl", "plotly")

install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

install_if_missing(required_packages)

library(shiny)
library(randomForest)
library(readxl)
library(plotly)

train_set2 <- read_excel("D:/steam/shiny1/train_set3.xlsx")

train_set2$Subgroup <- factor(train_set2$Subgroup, levels = c(0, 1), labels = c("HC", "HCC"))
train_set2$Sex <- factor(train_set2$Sex, levels = c(1, 0), labels = c("Male", "Female"))

rf_model <- randomForest(
  formula = Subgroup ~ .,
  data = train_set2,
  ntree = 500,
  mtry = 2,
  nodesize = 1,
  replace = TRUE,
  localImp = TRUE,
  nPerm = 1000
)

ui <- fluidPage(
  titlePanel(HTML("<span style='color:blue; font-size: 46px;'>F4-ASAD</span>_<span style='color:red; font-size: 46px;'>HCC</span>"))
  ,
  
  br(), br(),
  
  div(
    h4("Intended use:", style = "font-size: 20px;"),
    p("The F4-ASAD model is a validated diagnostic tool to calculate the probability of hepatocellular carcinoma (HCC) in individual patients.", style = "font-size: 18px;")
  ),
  
  sidebarLayout(
    sidebarPanel(
      h4("User Input Area", style = "font-size: 18px; font-weight: bold; margin-bottom: 15px;"),
      
      numericInput("age", "Age (years):", value = 30, min = 0, step = 1),
      selectInput("sex", "Sex:", choices = c("Male", "Female")),  
      numericInput("AFP", "AFP (Alpha-fetoprotein, ng/mL):", value = 10, min = 0, step = 0.01),
      numericInput("DCP", "DCP (Des-gamma carboxyprothrombin, ng/mL):", value = 1, min = 0, step = 0.01),
      actionButton("predict", "Calculate probability"),
      
      style = "padding: 15px; margin-top: 0px;"
    ),
    
    mainPanel(
      div(
        h4("Results:", style = "font-size: 18px;"),
        div(
          verbatimTextOutput("result", placeholder = TRUE),
          style = "font-size: 18px; font-weight: bold; margin-bottom: 20px;"
        ),
        plotlyOutput("gaugePlot"),
        h4("Interpretation:", style = "font-size: 18px;"),
        htmlOutput("interpretation", style = "font-size: 16px;")
      ),
      
      style = "padding-top: 15px; margin-top: -15px;"
    )
  )
)



server <- function(input, output) {
  
  observeEvent(input$predict, {
    new_data <- data.frame(
      Age = as.numeric(input$age),
      Sex = as.factor(input$sex),
      AFP = as.numeric(input$AFP),
      DCP = as.numeric(input$DCP)
    )
    
    new_data$Sex <- factor(new_data$Sex, levels = levels(train_set2$Sex))
    
    prob <- predict(rf_model, new_data, type = "prob")[, "HCC"]
    
    output$result <- renderText({
      paste("Predicted Probability of HCC:", round(prob, 3))
    })
    
    output$gaugePlot <- renderPlotly({
      plot_ly(
        type = "indicator",
        mode = "gauge+number",
        value = prob,
        number = list(valueformat = ".3f", font = list(size = 70, color = "black")),
        title = list(text = "", font = list(size = 18)),
        gauge = list(
          axis = list(
            range = list(0, 1), 
            tickfont = list(size = 16)  
          ),
          bar = list(color = "rgba(128,128,128,0.7)"),  
          steps = list(                     
            list(range = c(0, 0.5), color = "rgba(0,255,0,0.5)"),      
            list(range = c(0.5, 1), color = "rgba(255,0,0,0.5)")      
          )
        )
      )
    })
    
    output$interpretation <- renderUI({
      base_text <- "The calculated probability of developing HCC ranges from 0 to 1."
      likelihood_text <- "<b><span style='color:red;'>If the predicted probability is greater than 0.5, there is a likelihood of developing HCC.</span></b>"
      additional_text <- "As the predicted value gets closer to 1, the probability of developing HCC increases."
      
      HTML(paste(base_text, likelihood_text, additional_text))
    })
  })
}

shinyApp(ui = ui, server = server)
