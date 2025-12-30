
import delimited ../cleaned_data/ipeirotis_cleaned.csv, delim(",") clear
tempfile ipeirotis_cleaned
save `ipeirotis_cleaned'

****

use "../stata_output/all_residuals_cleaned.dta", replace

keep if data=="ipeirotis"

merge 1:1 group_id ///
	using `ipeirotis_cleaned', keepusing(description kw_parsed)
	
keep if _merge==3
drop _merge

export delimited "../stata_output/Monopsony_cleaned.csv", delim(";") replace
save "../stata_output/Monopsony_cleaned.dta", replace

***

clear
import delimited ///
	"/Users/kahrens/MyProjects/Dube_replication_condensed/ml_input/feature_ipeirotis_fullab.csv"
tempfile fullab
save `fullab'

clear
import delimited ///
	"/Users/kahrens/MyProjects/Dube_replication_condensed/ml_input/feature_ipeirotis_fullba.csv"
tempfile fullba
save `fullba'

use `fullab', clear
append using `fullba'
tempfile full
save `full'
export delimited "../stata_output/Monopsony_cleaned_all.csv", delim(";") replace
 