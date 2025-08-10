using Microsoft.AspNetCore.Identity;

namespace AuthService.Domain;

public sealed class UserPrincipal : IdentityUser
{
    public string WalletId { get; private set; }

    private UserPrincipal() { }

    public UserPrincipal(string walletId)
    {
        WalletId = walletId;
    }
}