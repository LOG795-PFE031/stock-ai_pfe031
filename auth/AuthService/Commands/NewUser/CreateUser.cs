using AuthService.Commands.Seedwork;

namespace AuthService.Commands.NewUser;

public sealed record CreateUser(string Username, string Password, string Role) : ICommand;