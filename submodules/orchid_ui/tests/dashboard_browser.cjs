// Legacy command retained for developer scripts; current dashboard regressions live here.
require('node:child_process').execFileSync(process.execPath,[require('node:path').join(__dirname,'two_page_browser.cjs'),...process.argv.slice(2)],{stdio:'inherit'});
