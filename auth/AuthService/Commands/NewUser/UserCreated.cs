using AuthService.Domain.Seedwork.Abstract;

namespace AuthService.Commands.NewUser;

public sealed class UserCreated : Event
{
    public string WalletId { get; init; } = string.Empty;
}