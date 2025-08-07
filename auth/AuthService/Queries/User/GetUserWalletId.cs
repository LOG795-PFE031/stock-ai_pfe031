using AuthService.Queries.Seedwork;

namespace AuthService.Queries.User;

public record GetUserWalletId(string Username) : IQuery;