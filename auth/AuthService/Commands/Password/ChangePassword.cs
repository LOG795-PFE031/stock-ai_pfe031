using AuthService.Commands.Seedwork;

namespace AuthService.Commands.Password;

public sealed record ChangePassword(string Username, string OldPassword, string NewPassword) : ICommand;