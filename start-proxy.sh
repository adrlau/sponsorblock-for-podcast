# nix-shell --run "mitmproxy --mode regular -s mitm.py --set console_eventlog_verbosity='warn'"

#add the proxy to your system or browser settings
#to make the proxy work with https sites go to http://mitm.it/ and install the certificate in your browser
nix-shell --run "mitmproxy --mode 'regular@8080'  -s mitm.py --showhost --set console_eventlog_verbosity='warn'"
