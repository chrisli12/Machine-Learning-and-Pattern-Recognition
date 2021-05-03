# EECS4404 : Machine Learning and Pattern Recognition <br/>
<h2> Group Project </h3>
<h3>Data Source : <a href ="https://archive.ics.uci.edu/ml/datasets/Bank+Marketing" target="_blank">UCI repository</a></h3>
<p> The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.</p>
<h3>Attribute information:</h3>
<h3>Input variables:</h3>
 <ol>
   # bank client data:
   <li>age (numeric)</li>
   <li>job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") </li>
   <li> marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)</li>
   <li> education (categorical: "unknown","secondary","primary","tertiary")</li>
   <li> default: has credit in default? (binary: "yes","no")</li>
   <li> balance: average yearly balance, in euros (numeric) </li>
   <li> housing: has housing loan? (binary: "yes","no")</li>
   <li>loan: has personal loan? (binary: "yes","no")</li>
   # related with the last contact of the current campaign:
  <li> contact: contact communication type (categorical: "unknown","telephone","cellular") </li>
  <li> day: last contact day of the month (numeric)</li>
  <li> month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")</li>
  <li> duration: last contact duration, in seconds (numeric)</li>
   # other attributes:
  <li>campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)</li>
  <li> pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)</li>
  <li> previous: number of contacts performed before this campaign and for this client (numeric)</li>
  <li> poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")</li>
  # Output variable (desired target):
  <li> y - has the client subscribed a term deposit? (binary: "yes","no")</li>
  </ol>
