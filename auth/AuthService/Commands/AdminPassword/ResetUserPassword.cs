using AuthService.Commands.Seedwork;

namespace AuthService.Commands.AdminPassword;

public sealed record ResetUserPassword(string Username, string Password) : ICommand;