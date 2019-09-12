# Store traffic monitor UI
This is a web-based UI specifically designed to work with the store-traffic-monitor reference implementation.   
The output of the Store-traffic-monitor reference implementation is read and displayed in real-time by the UI. For this reason, the UI and reference implementation must be running in parallel. 

**To avoid any browser issues, the application should be started first, and then the UI.**

## Running the UI
Go to UI directory present in store-traffic-monitor.
```
cd <path-to-store-traffic-monitor>/UI
```

### Install the dependencies
```
sudo apt install composer
composer install
```
Run the following command on the terminal to open the UI.<br>
Chrome*:
```
google-chrome  --user-data-dir=$HOME/.config/google-chrome/Store-traffic-monitor --new-window --allow-file-access-from-files --allow-file-access --allow-cross-origin-auth-prompt index.html
```
Firefox*:
```
firefox index.html
```
