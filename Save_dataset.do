
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


********************************************************************************
* save other numerical variables 
* both _fullab and _fullba are identical except from the top 100 word counts which I omit
********************************************************************************

clear
import delimited ///
	"/Users/kahrens/MyProjects/Dube_replication_condensed/ml_input/feature_ipeirotis_fullab.csv"
forvalues i = 0(1)99 {
	cap drop kw_d`i'
	cap drop desc_d`i'
	cap drop title_d`i'
	cap drop kw_r`i'
	cap drop desc_r`i'
	cap drop title_r`i'
}

tempfile fullab
local vlist

save `fullab'
export delimited "../stata_output/Monopsony_cleaned_all.csv", delim(";") replace
 
/*** the other file is identical; checking this below
clear
import delimited ///
	"/Users/kahrens/MyProjects/Dube_replication_condensed/ml_input/feature_ipeirotis_fullba.csv"
forvalues i = 0(1)99 {
	cap drop kw_d`i'
	cap drop desc_d`i'
	cap drop title_d`i'
	cap drop kw_r`i'
	cap drop desc_r`i'
	cap drop title_r`i'
}

tempfile fullba
foreach var of varlist duration-appr_num_gt {
	rename `var' ba_`var' 
}
save `fullba'

merge 1:1 group_id using `fullab'

di "`vlist'"
foreach var in `vlist' {
	di "`var'"
	assert `var'==ba_`var'
	//di r(N)
}
*/
