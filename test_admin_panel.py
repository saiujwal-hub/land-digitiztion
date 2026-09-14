import unittest
import dashboard_view
import auth_service
import accounts_store

class TestAdminPanel(unittest.TestCase):
    def test_render_admin_dashboard(self):
        page_bytes = dashboard_view.render_admin_dashboard(
            host_name="localhost:8001",
            colab_url="https://test-worker.ngrok-free.app",
            user_name="DevUser",
        )
        self.assertIsInstance(page_bytes, bytes)
        html_str = page_bytes.decode("utf-8")

        self.assertIn("Developer &amp; Admin <em>Console</em>", html_str)
        self.assertIn("Remote GPU Worker Status", html_str)

    def test_render_admin_users_page(self):
        page_bytes = dashboard_view.render_admin_users_page(
            host_name="localhost:8001",
            colab_url="https://test-worker.ngrok-free.app",
            user_name="DevUser",
        )
        self.assertIsInstance(page_bytes, bytes)
        html_str = page_bytes.decode("utf-8")

        self.assertIn("System Accounts <em>Directory</em>", html_str)
        self.assertIn("Registered System Users &amp; Officer Roster", html_str)
        self.assertIn("deleteUser", html_str)
        self.assertIn("resetUserDocs", html_str)

    def test_render_sidebar_admin_active(self):
        sidebar_html = dashboard_view._render_sidebar("admin", 2, 5, "Local CPU", role="admin")
        self.assertIn('href="/admin"', sidebar_html)
        self.assertIn('href="/admin/users"', sidebar_html)
        self.assertNotIn("Officer Workspace", sidebar_html)
        self.assertNotIn("User Desk", sidebar_html)

    def test_render_sidebar_user_officer_hidden(self):
        user_sidebar = dashboard_view._render_sidebar("dashboard", 2, 5, "Local CPU", role="user")
        officer_sidebar = dashboard_view._render_sidebar("dashboard", 2, 5, "Local CPU", role="officer")
        self.assertNotIn('href="/admin"', user_sidebar)
        self.assertNotIn('href="/admin"', officer_sidebar)

    def test_delete_user_account(self):
        # 1. Master admin cannot be deleted
        ok, msg = auth_service.delete_user_account("usr_admin_master")
        self.assertFalse(ok)
        self.assertIn("master administrator account", msg)

        # 2. Create temporary user and delete
        temp_user = accounts_store.create_user("Temp Delete Target", "user", [{"type": "email", "identifier": "tempdel@test.com"}])
        temp_id = temp_user["user_id"]
        ok, msg = auth_service.delete_user_account(temp_id)
        self.assertTrue(ok)
        self.assertIsNone(accounts_store.get_user(temp_id))

if __name__ == "__main__":
    unittest.main()

