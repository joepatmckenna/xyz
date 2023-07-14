export async function load({ params }) {
	const post = await import(`../${params.post}.md`);
	const { title, date } = post.metadata;
	const Content = post.default;

	return {
		title,
		date,
		Content
	};
}
