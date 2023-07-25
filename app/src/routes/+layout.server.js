export const load = async ({ cookies }) => {
	return { darkMode: cookies.get('darkMode') || false };
};
