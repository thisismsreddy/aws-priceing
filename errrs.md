Thought for a couple of seconds


Here’s a battle-tested pattern for running your Flask app (let’s call it `app4.py`) behind Nginx on RHEL 9:

---

## 1. Install prerequisites

```bash
sudo dnf install -y python3 python3-venv nginx
```

If you haven’t already, create a Python virtualenv and install your requirements:

```bash
cd /home/srini/aws-priceing
python3 -m venv venv
source venv/bin/activate
pip install flask openstacksdk pandas gunicorn
deactivate
```

---

## 2. Create a `gunicorn` Systemd service

Create `/etc/systemd/system/pricing.service`:

```ini
[Unit]
Description=AWS Pricing Comparison Flask App
After=network.target

[Service]
User=srini
Group=nginx
WorkingDirectory=/home/srini/aws-priceing
Environment="PATH=/home/srini/aws-priceing/venv/bin"
ExecStart=/home/srini/aws-priceing/venv/bin/gunicorn \
    --workers 3 \
    --bind unix:/run/pricing.sock \
    app4:app

[Install]
WantedBy=multi-user.target
```

* **User/Group**: run as your user (`srini`) but group `nginx` so Nginx can read the socket.
* **socket**: we bind to a unix socket at `/run/pricing.sock`.

Enable & start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pricing.service
```

Check it’s listening:

```bash
sudo ss -lxn | grep /run/pricing.sock
```

---

## 3. Open SELinux & firewall

Allow Nginx to talk to that socket under SELinux:

```bash
sudo setsebool -P httpd_can_network_connect on
```

Open HTTP in the firewall:

```bash
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --reload
```

---

## 4. Configure Nginx as a reverse proxy

Create `/etc/nginx/conf.d/pricing.conf`:

```nginx
server {
    listen       80;
    server_name  pricing.example.com;    # your FQDN or IP

    # serve static files directly if any
    root /home/srini/aws-priceing/static;
    location /static/ {
        alias /home/srini/aws-priceing/static/;
        expires 30d;
    }

    location / {
        # proxy to gunicorn socket
        include         proxy_params;
        proxy_pass      http://unix:/run/pricing.sock;
    }
}
```

Test & reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 5. Verify

* Browse to `http://pricing.example.com/` (or your server IP).
* You should see your Flask form served by Nginx → Gunicorn → Flask.

---

### Tips

* **SSL**: Once it’s working over HTTP, plug in Certbot or your corporate TLS cert in `/etc/nginx/conf.d/pricing.conf`.
* **Logging**: Gunicorn logs to `journald` by default; you can add `--access-logfile - --error-logfile -` to the `ExecStart` if you want Nginx-style logs.
* **Scaling**: Tune the `--workers` count to roughly `(2 × CPU cores) + 1`.

That’s it—now you’ve got a production-ready RHEL 9 stack with Nginx fronting your Flask pricing app!
















*

Serving Flask app 'app4'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 113-789-396
127.0.0.1 - - [05/May/2025 22:12:54] "GET / HTTP/1.1" 500 -
Traceback (most recent call last):
  File "/home/srini/env/lib/python3.8/site-packages/flask/app.py", line 1498, in __call__
    return self.wsgi_app(environ, start_response)
  File "/home/srini/env/lib/python3.8/site-packages/flask/app.py", line 1476, in wsgi_app
    response = self.handle_exception(e)
  File "/home/srini/env/lib/python3.8/site-packages/flask/app.py", line 1473, in wsgi_app
    response = self.full_dispatch_request()
  File "/home/srini/env/lib/python3.8/site-packages/flask/app.py", line 882, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "/home/srini/env/lib/python3.8/site-packages/flask/app.py", line 880, in full_dispatch_request
    rv = self.dispatch_request()
  File "/home/srini/env/lib/python3.8/site-packages/flask/app.py", line 865, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)  # type: ignore[no-any-return]
  File "/home/srini/aws-priceing/app4.py", line 364, in index
    return render_template_string(FORM_HTML,
  File "/home/srini/env/lib/python3.8/site-packages/flask/templating.py", line 162, in render_template_string
    return _render(app, template, context)
  File "/home/srini/env/lib/python3.8/site-packages/flask/templating.py", line 131, in _render
    rv = template.render(context)
  File "/home/srini/env/lib/python3.8/site-packages/jinja2/environment.py", line 1301, in render
    self.environment.handle_exception()
  File "/home/srini/env/lib/python3.8/site-packages/jinja2/environment.py", line 936, in handle_exception
    raise rewrite_traceback_stack(source=source)
  File "<template>", line 17, in top-level template code
TypeError: url_for() takes 2 positional arguments but 3 were given


<img src="{{ url_for('static', filename='loading.gif') }}" alt="Loading..." />
